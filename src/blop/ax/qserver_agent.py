import logging

###
import threading
import uuid
from collections.abc import Mapping, Sequence, Callable
from typing import Any, Concatenate, Literal, ParamSpec

import numpy as np
import pandas as pd
from bluesky.callbacks import CallbackBase
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.zmq import REManagerAPI
from numpy.typing import ArrayLike
from ophyd import Signal
from tiled.queries import Eq
from ax.api.types import TOutcome, TParameterization

from blop import DOF, Objective
from blop.ax.agent import Agent as BlopAxAgent  # type: ignore[import-untyped]

import databroker
from ax import Client
from bluesky.plans import PerStep
from bluesky.protocols import Readable
from databroker import Broker
from tiled.client.container import Container

from ..data_access import DatabrokerDataAccess, TiledDataAccess
from ..digestion_function import default_digestion_function
from .dof import DOF, DOFConstraint
from ..objectives import Objective
from ..protocols import AcquisitionPlan, EvaluationFunction, OptimizationProblem, Sensor
logger = logging.getLogger("blop")


class ConsumerCallback(CallbackBase):
    """
    A child of Callback base which caches the start document and calls a callback function on the stop document

    """

    def __init__(self, callback: callable = None, enable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_doc_cache = None
        self.callback = callback  # A function that is called when stop is called
        self.enable = enable

    def start(self, doc):
        if self.enable:
            self.start_doc_cache = doc

    def stop(self, doc):
        if self.enable:
            self.callback(self.start_doc_cache, doc)
            self._clear_cache()

    def _clear_cache(self):
        self.start_doc_cache = None


class ZMQConsumer:
    """
    Allows us to start a thread which will listen to docs and call the callback in CallbackBase
    """

    def __init__(self, zmq_consumer_ip_address, zmq_consumer_port, callback: callable = None):
        self.zmq_consumer_ip_address = zmq_consumer_ip_address
        self.zmq_consumer_port = zmq_consumer_port

        self.zmq_consumer = RemoteDispatcher(f"{self.zmq_consumer_ip_address}:{self.zmq_consumer_port}")
        self.zmq_consumer_callback = ConsumerCallback(callback=callback, enable=True)
        self.zmq_consumer.subscribe(self.zmq_consumer_callback)
        self._zmq_thread = None

    def start_zmq_listener_thread(self):
        logger.info(f"Starting ZMQ Callback Thread")
        
        self._zmq_thread = threading.Thread(target=self.zmq_consumer.start, name="zmq-consumer", daemon=True)
        self._zmq_thread.start()

P = ParamSpec("P")
DigestionFunction = Callable[Concatenate[int, dict[str, list[Any]], P], TOutcome]

class BlopQserverAgent(BlopAxAgent):
    
    """
    An agent interface that uses Ax as the backend for optimization and experiment tracking.
    
    This agent connects to a Qserver to submit plans, rather than emmitting messages to be consumed directly

    Attributes
    ----------
    readables : list[Readable]
        The readables to use for acquisition. These should be the minimal set
        of readables that are needed to compute the objectives.
    dofs : list[DOF]
        The degrees of freedom that the agent can control, which determine the output of the model.
    objectives : list[Objective]
        The objectives which the agent will try to optimize.
    db : Broker | Container
        The databroker or tiled instance to read back data from a Bluesky run.
    dof_constraints : Sequence[DOFConstraint], optional
        Constraints on DOFs to refine the search space.
    digestion : DigestionFunction
        The function to produce objective values from a dataframe of acquisition results.
    digestion_kwargs : dict
        Additional keyword arguments to pass to the digestion function.
        
    Example
    ---------
    
    dofs = [
        DOF(name='motor1', search_domain=(-6.0, 6.0)),
        DOF(name='motor2' ,search_domain=(-6.0, 6.0)),
    
    ]

    objectives = [
        Objective(name="himmel_det", transform = 'log',target="min"),
    
    ]

    readables = ['himmel_det']
    
    agent = BlopQserverAgent(
                readables=readables,
                dofs=dofs,
                qserver_control_addr="tcp://localhost:60615",
                qserver_info_addr="tcp://localhost:60625",
                zmq_consumer_ip= "localhost",
                zmq_consumer_port= "5578",
                objectives=objectives,
                acquisition_plan= "acquire",
                db=db,
            )
            
    
    agent.RM.environment_open()

    agent.optimize(iterations=30, n_points=1)
    """

    def __init__(
        self,
        readables: Sequence[Readable],
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        db: Broker | Container,
        qserver_control_addr=None,
        qserver_info_addr:str=None,
        zmq_consumer_ip:str=None,
        zmq_consumer_port:str=None,
        acquisition_plan: str = "acquire",
        dof_constraints: Sequence[DOFConstraint] = None,
        digestion: DigestionFunction = default_digestion_function,
        digestion_kwargs: dict | None = None,
        evaluation:EvaluationFunction=None
    ):
        
        super().__init__(
            sensors=readables,
            dofs=dofs,
            objectives=objectives,
            evaluation=evaluation,
            dof_constraints=dof_constraints,
            acquisition_plan=acquisition_plan
        )
        
        #self.client = Client()
        self.digestion = digestion
        self.digestion_kwargs = digestion_kwargs or {}
        
        
        # I store this for data access although I need to check which version of Tiled I am using
        self.db = db
                
        # Instantiate an object that can communicate with the queueserver
        self.RM = REManagerAPI(zmq_control_addr=qserver_control_addr, zmq_info_addr=qserver_info_addr)  # To Do, Add arguements to class init

        # Should plans be submitted and automatically started, or not?
        self._queue_autostart = False

        # Instantiate an object that will listen for start and stop documents and call a method on the stop document
        self.zmq_consumer = ZMQConsumer(zmq_consumer_ip, zmq_consumer_port, callback=self._stop_doc_callback)
        self.zmq_consumer.start_zmq_listener_thread()

        # Should we do something when there is a new event?
        self._listen_to_events = True

        # Should new suggestions be made automatically until all of the trials are complete?
        self.learn_continuous_suggestion = True

        # Learning parameters
        self.learn_num_itterations = 30
        self.learn_current_itteration = 0
        self.learn_n_points = 1
        self.learn_route = True
        self.agent_suggestion_uid = None
        self.trials = None
        self.acquisition_finished = False
        self.optimization_problem = None

    
    def _stop_doc_callback(self, start_doc, stop_doc):
        """
        In here we can decide whether our experiment requested has completed

        If it has completed, we can digest the data from it and move on to the next point.
        """
        
        if self._listen_to_events:
            self.acquisition_finished = True
            logger.info("Stop Document found")
            
            # We can't just take any stop document, there could be other things in the queue, so we search for it
            
            # get the (list of) run which contains the data we want. We can get this by calling search
            results_db = self.db.search(Eq("agent_suggestion_uid", self.agent_suggestion_uid))
            
            # Check that there is 1 and only one entry
            if len(results_db) == 1:
  
 
                results_run =  results_db[-1]
                uid = results_run.metadata['start']['uid']
                              
                outcomes = self.optimization_problem.evaluation_function(uid, self.trials)
                
                # Learn from the data
                self.optimization_problem.optimizer.ingest(outcomes)

                # After this is complete, call gen_next_trials again if required
                if self.learn_continuous_suggestion:
                    if self.learn_current_itteration < self.learn_num_itterations:
                        logger.info("making another suggestion")
                        self.suggest()
                    else:
                        self.learn_current_itteration = 0
                        logger.info("made all required suggestions")
        
    def optimize(self, iterations = 1, n_points = 1):
        
        """
        This method will create the optimization problem, suggest points and execute them in the QS
        """
        
        self.optimization_problem = self.to_optimization_problem()
        self.learn_num_itterations = iterations
        self.n_points = n_points
              
        self.suggest()
         
    def suggest(self):
        """
        ask for suggestions, then send the values to the queueserver (like one half of learn)

        """

        # record this itteration
        self.learn_current_itteration = self.learn_current_itteration + 1
        
        # Get the trials to perform     
        self.trials = self._optimizer._client.get_next_trials(self.learn_n_points)
        
        # acquire the values from those trials
        logger.info("sending suggestion to acquire")
        self.agent_suggestion_uid = self.acquire(self.trials)

    def acquire(self, trials: dict[int, TParameterization], per_step: PerStep | None = None):
        """Acquire and digest according to the acquisition and digestion plans.

        This method will send a plan to the queueserver and block until the stop document of that plan is recieved


        Parameters
        ----------
        trials : dict[int, TParameterization]
            A dictionary mapping trial indices to their suggested parameterizations.
        """

        try:
            self.acquisition_finished = False

            # Create a unique identifier which will connect the children to the parent batch
            # This batch ID will be used by all runs from this request by the agent. It will be used by the digestion function later to work out what happened.

            agent_suggestion_uid = str(uuid.uuid4())
            kwargs = {}
            kwargs.setdefault("md", {})
            kwargs["md"]["agent_suggestion_uid"] = agent_suggestion_uid
            suggestions = [{ "_id": trial_index, **parameterization, }  for trial_index, parameterization in trials.items()  ]
            kwargs['md']["blop_suggestions"]= suggestions
        
            # ensure the values of inputs are float 64, since weirdly float 32 cannot be serialized by json to be sent to qserver
            item = BPlan(
                self.acquisition_plan,
                readables = self.sensors,
                dofs = [dof.name for dof in self.dofs],
                trials = trials,          
                md=kwargs["md"],
            )

            # Send the plan to the Run Engine Manager
            r = self.RM.item_add(item)
            logger.debug(
                f"Sent http-server request for trials {trials} with agent_suggestion_uid= {agent_suggestion_uid}\n.Received reponse: {r}"
            )

            # If the queue should start automatically, then start the queue.
            if self._queue_autostart:
                logger.debug("Waiting for Queue to be idle or paused")
                self.RM.wait_for_idle_or_paused(timeout=600)
                r = self.RM.queue_start()
                logger.debug(f"Sent http-server request to start the queue\n.Received reponse: {r}")

        except KeyboardInterrupt as interrupt:
            raise interrupt

        return agent_suggestion_uid



  
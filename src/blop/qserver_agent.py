from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from bluesky_adaptive.agents.base import Agent as BlueskyAdaptiveBaseAgent  # type: ignore[import-untyped]
from bluesky_adaptive.typing import BlueskyRunLike  # type: ignore[import-untyped]
from numpy.typing import ArrayLike

from blop.agent import BaseAgent as BlopAgent  # type: ignore[import-untyped]
from blop.agent import Agent as BlopFullAgent  # type: ignore[import-untyped]
from blop.digestion import default_digestion_function  # type: ignore[import-untyped]

from blop import DOF, Objective
import pandas as pd
from numpy.typing import ArrayLike
from collections.abc import Callable, Mapping, Sequence
from bluesky_queueserver_api.zmq import REManagerAPI
from bluesky_queueserver_api import BPlan
from ophyd import Signal
import time
from typing import Any
import uuid
from tiled.queries import Eq

### 
import threading
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky.callbacks import CallbackBase
import logging

logger = logging.getLogger("blop")

class ConsumerCallback(CallbackBase):
    
    
    """
    A child of Callback base which caches the start document and calls a callback function on the stop document
    
    """
    
    def __init__(self,callback:callable=None,enable=True,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.start_doc_cache = None
        self.callback = callback # A function that is called when stop is called
        self.enable = enable
        
    def start(self,doc):
        if self.enable:
            self.start_doc_cache = doc
        
    def stop(self,doc):
        if self.enable:
            self.callback(self.start_doc_cache,doc)
            self._clear_cache()
            
    def _clear_cache(self):
        self.start_doc_cache = None
        

class ZMQConsumer:
    
    """
    Allows us to start a thread which will listen to docs and call the callback in CallbackBase
    """
    def __init__(self,zmq_consumer_ip_address,zmq_consumer_port,callback:callable=None):
        
        self.zmq_consumer_ip_address = zmq_consumer_ip_address
        self.zmq_consumer_port = zmq_consumer_port
        
        self.zmq_consumer = RemoteDispatcher(f"{self.zmq_consumer_ip_address}:{self.zmq_consumer_port}")
        self.zmq_consumer_callback = ConsumerCallback(callback=callback, enable=True)
        self.zmq_consumer.subscribe(self.zmq_consumer_callback)
        self._zmq_thread = None
        
    def start_zmq_listener_thread(self):
        print('Starting ZMQ Callback Thread')
        self._zmq_thread = threading.Thread(target=self.zmq_consumer.start, name="zmq-consumer", daemon=True)
        self._zmq_thread.start()
        






class BlopQserverAgent(BlopFullAgent):
    
    def __init__(
        self,
        dofs: Sequence[DOF],
        objectives: Sequence[Objective],
        db = None,
        detectors: Sequence[Signal] | None = None,
        acquisition_plan: str = 'list_scan_2d',
        verbose: bool = False,
        enforce_all_objectives_valid: bool = True,
        exclude_pruned: bool = True,
        model_inactive_objectives: bool = False,
        tolerate_acquisition_errors: bool = False,
        sample_center_on_init: bool = False,
        trigger_delay: float = 0,
        train_every: int = 3,
    ):
        """
        A Bayesian optimization agent.

        Parameters
        ----------
        dofs : Sequence[DOF]
            The degrees of freedom that the agent can control, which determine the output of the model.
        objectives : Sequence[Objective]
            The objectives which the agent will try to optimize.
        acquisition_plan : Callable, optional
            A plan that samples the beamline for some given inputs, by default default_acquisition_plan.
            Called directly in Agent, used only by ``__name__`` in BlueskyAdaptiveAgent.
            Default: ``default_acquisition_plan``
        db : tiled_client containing measurement results
        verbose : bool, optional
            To be verbose or not, by default False
            Default: False
        enforce_all_objectives_valid : bool, optional
            Whether the agent should exclude from fitting points with one or more invalid objectives.
            Default: True
        exclude_pruned : bool, optional
            Whether to exclude from fitting points that have been pruned after running agent.prune().
            Default: True
        model_inactive_objectives : bool, optional
            Whether the agent should update models for outcomes that affect inactive objectives.
            Default: False
        tolerate_acquisition_errors : bool, optional
            Whether to allow errors during acquistion. If `True`, errors will be caught as warnings.
            Default: False
        sample_center_on_init : bool, optional
            Whether to sample the center of the DOF limits when the agent has no data yet.
            Default: False
        train_every : int, optional
            How many samples to take before retraining model hyperparameters.
            Default: 4
            
            
        ------
        dofs = [
            DOF(description="motor1", name='motor1', search_domain=(-6.0, 6.0)),
            DOF(description="motor2", name='motor2' ,search_domain=(-6.0, 6.0)),
        
        ]

        objectives = [
            Objective(name="himmel_det", transform = 'log',target="min"),
        
        ]

        agent = BlopQserverAgent(
            dofs=dofs,
            objectives=objectives,
            detectors=['himmel_det'],
            verbose=True,
            db=db,
            tolerate_acquisition_errors=False,
            enforce_all_objectives_valid=True,
            train_every=1,
        )      
        """

        super().__init__(
            dofs=dofs,
            objectives=objectives,
            db = db,
            acquisition_plan=acquisition_plan,
            verbose=verbose,
            enforce_all_objectives_valid=enforce_all_objectives_valid,
            exclude_pruned=exclude_pruned,
            model_inactive_objectives=model_inactive_objectives,
            tolerate_acquisition_errors=tolerate_acquisition_errors,
            sample_center_on_init=sample_center_on_init,
            trigger_delay=trigger_delay,
            train_every=train_every,
            detectors = detectors
            
        )
        
        # Instantiate an object that can communicate with the queueserver
        self.RM = REManagerAPI() #To Do, Add arguements to class init
        
        # Should plans be submitted and automatically started, or not?
        self._queue_autostart = False
        
        # Instantiate an object that will listen for start and stop documents and call a method on the stop document
        self.zmq_consumer = ZMQConsumer('localhost','5578',callback=self._stop_doc_callback)
        self.zmq_consumer.start_zmq_listener_thread()
        
        # Should we do something when there is a new event?
        self._listen_to_events= True
        
        self.acquisition_finished = False
        self.current_acqf_name = None
        
        self.learn_continuous_suggestion = True
        
        # Learning parameters
        self.learn_num_itterations = 1
        self.learn_acq_func = 'qr'
        self.learn_current_itteration = 0
        self.learn_n_points = 36
        self.learn_upsample = 1
        self.learn_route = True
        self.agent_suggestion_uid = None
        self.learn_force_train = False
        self.learn_append = True
        
        
    def _stop_doc_callback(self,start_doc, stop_doc):
        """
        In here we can decide whether our experiment requested has completed
        
        If it has completed, we can digest the data from it and move on to the next point. 
        """
        
        """
        We need to check that both:
        
        1. The stop document for the plan we submitted is recieved
        2. The stop document of any child plans (run on other queueservers) is also recieved
        
        """
        if self._listen_to_events:
            self.acquisition_finished = True
            self.ingest()
            print(f"Stop Document found, not yet checking if it's the right one!")
            
    def suggest(self):
        
        """
        ask for suggestions, then send the values to the queueserver (like one half of learn)
        
        """
        
            
        # record this itteration
        self.learn_current_itteration = self.learn_current_itteration + 1
        
        if self.verbose:
            logger.info(f"running iteration {self.learn_current_itteration} / {self.learn_num_itterations}")
            
        for single_acqf in np.atleast_1d(self.learn_acq_func):
            res = self.ask(n=self.learn_n_points, acqf=single_acqf, upsample=self.learn_upsample, route=self.learn_route)
            
            #acquire new table from the experiment. This function is blocking. 
            self.current_acqf_name = res["acqf_name"]
            print(f"sending suggestion to acquire")
            self.agent_suggestion_uid = self.acquire(res["points"])
               

        
    def acquire(self, points: dict[str, list[ArrayLike]]):
        """Acquire and digest according to the self's acquisition and digestion plans.
        
        This method will send a plan to the queueserver and block until the stop document of that plan is recieved
        

        Parameters
        ----------
        acquisition_inputs :
            A 2D numpy array comprising inputs for the active and non-read-only DOFs to sample.
        """

        if self.db is None:
            raise ValueError("Cannot run acquistion without databroker instance!")

        acquisition_dofs = self.dofs(active=True, read_only=False)
        for dof in acquisition_dofs:
            if dof.name not in points:
                raise ValueError(f"Cannot acquire points; missing values for {dof.name}.")

        n = len(points[dof.name])

        try:
            agent_suggestion_uid = self.send_acquisiton_plan(
                acquisition_dofs,
                points,
                [*self.detectors, *self.dofs.devices],
                
            )
            
        except KeyboardInterrupt as interrupt:
            raise interrupt
        
        return agent_suggestion_uid
    
    def unpack_run(self, runs) -> pd.DataFrame:
    
        #Put the code in here to create a pandas dataframe from the input data in our run(s).
        # We can look in the first run to get data from others. 
        
        print(f"Parsing runs and reading dataframe")
        df = runs[-1].primary.read().to_dataframe()
        return df
              
    def ingest(self):
        
        """
        this gets run when we get a new valid stop document
        """       
        results_db = self.db.search(Eq("agent_suggestion_uid", self.agent_suggestion_uid))
            
        new_table = self.unpack_run(results_db)

        new_table.loc[:, "acqf"] = self.current_acqf_name

        x = {key: new_table.loc[:, key].tolist() for key in self.dofs.names}
        y = {key: new_table.loc[:, key].tolist() for key in self.objectives.names}
        metadata = {
            key: new_table.loc[:, key].tolist() for key in new_table.columns if (key not in x) and (key not in y)
        }
        self.tell(x=x, y=y, metadata=metadata, append=self.learn_append, force_train=self.learn_force_train)
        
        if self.learn_continuous_suggestion:
        
            if self.learn_current_itteration < self.learn_num_itterations:
                print(f"making another suggestion")
                self.suggest()
            else:
                 self.learn_current_itteration = 0
        
            
        
    
    def send_acquisiton_plan(self, dofs: Sequence[DOF], inputs: Mapping[str, Sequence[Any]], dets: Sequence[Signal], **kwargs: Any)->str:
        
        """
        sends a plan to the queueserver, returns the uid of that plan
        
        sets the self.acquisition_finished = False
        """
        self.acquisition_finished = False
        
        # Create a unique identifier which will connect the children to the parent batch
        # This batch ID will be used by all runs from this request by the agent. It will be used by the digestion function later to work out what happened. 

        batch_id = str(uuid.uuid4())
        kwargs.setdefault("md",{})
        kwargs["md"]["agent_suggestion_uid"] = batch_id
        kwargs["md"]["acqf_name"] = self.current_acqf_name
    
        # ensure the values of inputs are float 64, since weirdly float 32 cannot be serialized by json to be sent to qserver
        
        positions1 = [ float(x) if isinstance(x, np.float32) else x for x in inputs["motor1"] ]
        positions2 = [ float(x) if isinstance(x, np.float32) else x for x in inputs["motor2"] ]
        
        item = BPlan(self.acquisition_plan,self.detectors,motor1='motor1', positions1=positions1,motor2="motor2", positions2=positions2, md =kwargs['md'])

        
        r = self.RM.item_add(item)
        logger.debug(f"Sent http-server request for points {inputs} with agent_suggestion_uid= {batch_id}\n." f"Received reponse: {r}")
        
        #If the queue should start automatically, then start the queue.
        if self._queue_autostart:
            logger.debug(f"Waiting for Queue to be idle or paused")
            self.RM.wait_for_idle_or_paused(timeout=600)
            r= self.RM.queue_start()
            logger.debug(f"Sent http-server request to start the queue\n." f"Received reponse: {r}")
            
        return batch_id
                

 


class BlueskyAdaptiveAgent(BlueskyAdaptiveBaseAgent, BlopAgent):
    """A BlueskyAdaptiveAgent that uses Blop for the underlying agent."""

    # TODO: Move into main package once databroker V2 is supported

    def __init__(
        self,
        *,
        acqf_string: str,
        route: bool,
        sequential: bool,
        upsample: int,
        acqf_kwargs: dict[str, Any],
        detector_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._acqf_string = acqf_string
        self._route = route
        self._sequential = sequential
        self._upsample = upsample
        self._acqf_kwargs = acqf_kwargs
        self._detector_names = detector_names or []

    @property
    def detector_names(self) -> list[str]:
        return [str(name) for name in self._detector_names]

    @detector_names.setter
    def detector_names(self, names: list[str]):
        self._detector_names = list(names)

    @property
    def acquisition_function(self) -> str:
        return str(self._acqf_string)

    @acquisition_function.setter
    def acquisition_function(self, acqf_string: str):
        self._acqf_string = str(acqf_string)

    @property
    def route(self) -> bool:
        return bool(self._route)

    @route.setter
    def route(self, route: bool):
        self._route = route

    @property
    def sequential(self) -> bool:
        return bool(self._sequential)

    @sequential.setter
    def sequential(self, sequential: bool):
        self._sequential = sequential

    @property
    def upsample(self) -> int:
        return int(self._upsample)

    @upsample.setter
    def upsample(self, upsample: int):
        self._upsample = int(upsample)

    @property
    def acqf_kwargs(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self._acqf_kwargs.items()}

    def update_acqf_kwargs(self, **kwargs):
        self._acqf_kwargs.update(kwargs)

    def server_registrations(self) -> list[str]:
        """This is how we make these avaialble to the REST API."""
        self._register_method("Update Acquistion Function Kwargs", self.update_acqf_kwargs)
        self._register_property("Acquisition Function", self.acquisition_function, self.acquisition_function)
        self._register_property("Route Points", self.route, self.route)
        self._register_property("Sequential Points", self.sequential, self.sequential)
        self._register_property("Upsample Points", self.upsample, self.upsample)
        return super().server_registrations()

    def ask(self, batch_size: int) -> tuple[Sequence[dict[str, ArrayLike]], Sequence[ArrayLike]]:
        default_result = super().ask(
            n=batch_size,
            acqf=self._acqf_string,
            route=self._route,
            sequential=self._sequential,
            upsample=self._upsample,
            **self._acqf_kwargs,
        )

        """res = {
            "points": {dof.name: list(points[..., i]) for i, dof in enumerate(active_dofs(read_only=False))},
            "acqf_name": acqf_config["name"],
            "acqf_obj": list(np.atleast_1d(acqf_obj.numpy())),
            "acqf_kwargs": acqf_kwargs,
            "duration_ms": duration,
            "sequential": sequential,
            "upsample": upsample,
            "read_only_values": read_only_values,
            # "posterior": p,
        }
        """

        points: dict[str, list[ArrayLike]] = default_result.pop("points")
        acqf_obj: list[ArrayLike] = default_result.pop("acqf_obj")
        # Turn dict of list of points into list of consistently sized points
        points: list[tuple[ArrayLike]] = list(zip(*[value for _, value in points.items()], strict=False))
        dicts = []
        for point, obj in zip(points, acqf_obj, strict=False):
            d = default_result.copy()
            d["point"] = point
            d["acqf_obj"] = obj
            dicts.append(d)
        return points, dicts

    def tell(self, x: dict[str, ArrayLike], y: dict[str, ArrayLike]):
        x = {key: x_i for x_i, key in zip(x, self.dofs.names, strict=False)}
        y = {key: y_i for y_i, key in zip(y, self.objectives.names, strict=False)}
        super().tell(data={**x, **y})
        return {**x, **y}

    def report(self) -> dict[str, Any]:
        raise NotImplementedError("Report is not implmented for BlueskyAdaptiveAgent")

    def unpack_run(self, run: BlueskyRunLike) -> tuple[list[ArrayLike], list[ArrayLike]]:
        """Use my DOFs to convert the run into an independent array, and my objectives to create the dependent array.
        In practice for shape management, we will use lists not np.arrays at this stage.
        Parameters
        ----------
        run : BlueskyRun

        Returns
        -------
        independent_var :
            The independent variable of the measurement
        dependent_var :
            The measured data, processed for relevance
        """
        if not self.digestion or self.digestion == default_digestion_function:
            # Assume all raw data is available in primary stream as keys
            return (
                [run.primary.data[key].read() for key in self.dofs.names],
                [run.primary.data[key].read() for key in self.objectives.names],
            )
        else:
            # Hope and pray that the digestion function designed for DataFrame can handle the XArray
            data: pd.DataFrame = self.digestion(run.primary.data.read(), **self.digestion_kwargs)
            return [data.loc[:, key] for key in self.dofs.names], [data.loc[:, key] for key in self.objectives.names]

    def measurement_plan(self, point: ArrayLike) -> tuple[str, list[Any], dict[str, Any]]:
        """Fetch the string name of a registered plan, as well as the positional and keyword
        arguments to pass that plan.

        Args/Kwargs is a common place to transform relative into absolute motor coords, or
        other device specific parameters.

        By default, this measurement plan attempts to use in the built in functionality in a QueueServer compatible way.
        Signals and Devices are not passed as objects, but serialized as strings for the RE as a service to use.

        Parameters
        ----------
        point : ArrayLike
            Next point to measure using a given plan

        Returns
        -------
        plan_name : str
        plan_args : List
            List of arguments to pass to plan from a point to measure.
        plan_kwargs : dict
            Dictionary of keyword arguments to pass the plan, from a point to measure.
        """
        if isinstance(self.acquisition_plan, Callable):
            plan_name = self.acquisition_plan.__name__
        else:
            plan_name = self.acquisition_plan
        if plan_name == "default_acquisition_plan":
            # Convert point back to dict form for the sake of compatability with default plan
            acquisition_dofs = self.dofs(active=True, read_only=False)

            return self.acquisition_plan.__name__, [
                acquisition_dofs,
                {dof.name: point[i] for i, dof in enumerate(acquisition_dofs)},
                [*self.detector_names, *[dev.__name__ for dev in self.dofs.devices]],
            ]
        else:
            raise NotImplementedError("Only default_acquisition_plan is implemented")


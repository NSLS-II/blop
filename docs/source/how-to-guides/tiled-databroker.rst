Tiled/Databroker with Blop
================================
This guide explains how we can use Tiled and Databroker for data storage and retrieval with Blop.

Setting Up Data Access
-----------------------

To access the data for optimization, you have to connect to a Tiled server or Databroker instance:

**Tiled:**

.. code-block:: python

    from tiled.client import from_uri, from_profile
    
    # Connect via profile (if available)
    tiled_client = from_profile("profile_name")
    
    # Connect via URI
    tiled_client = from_uri("uri_string")

**Databroker:**

.. code-block:: python

    from databroker import Broker
    db = Broker.named("profile_name")

For more details, see `Tiled <https://github.com/bluesky/tiled>`_ or `Databroker <https://github.com/bluesky/databroker>`_ .

Data Storage with Blop's Default Plans
---------------------------------------

Blop provides a default acquisition plan (:func:`blop.plans.default_acquire`) that handle data acquisition. This plan:

- Uses the **"primary" stream** to store all acquired data
- Includes a default metadata key **blop_suggestion_ids** which identifies suggestions that were acquired at each step of the scan.

When a custom acquisition plan is used, how the data is stored depends on the plan implementation. 

Creating an Evaluation Function
--------------------------------

To access data from Tiled or Databroker within your evaluation function, create a class that:

1. Accepts a client/broker instance in its ``__init__`` method
2. Implements a ``__call__`` method that retrieves data using the latest run UID
3. Processes the data to compute optimization objectives

Evaluation Function with Tiled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an example evaluation function that reads data from Tiled for where all data is stored in the "primary" stream:

.. code-block:: python

    from tiled.client.container import Container

    class MyEvaluation:
        def __init__(self, tiled_client: Container):
            self.tiled_client = tiled_client

        def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
            # Access the data of a specific run
            run = self.tiled_client[uid]
            
            # Read data for specific signals from the "primary" stream
            x1_data = run["primary/x1"].read()
            
            # Process each suggestion
            outcomes = []
            for suggestion in suggestions:
                suggestion_id = suggestion["_id"]
                x1 = x1_data[suggestion_id % len(x1_data)]
                outcome = {
                    "_id": suggestion["_id"],
                    "objective1": 0.1,
                    "objective2": 0.2,
                }
                outcomes.append(outcome)
            return outcomes
            
Evaluation Function with Databroker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an equivalent evaluation function using Databroker:

.. code-block:: python

    class MyEvaluation:
        def __init__(self, db):
            self.db = db

        def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
            # Access the run data as a pandas DataFrame
            run = self.db[uid].table()
            
            # Extract data columns
            x1_data = run["x1"]
            
            # Process each suggestion
            outcomes = []
            for suggestion in suggestions:
                suggestion_id = suggestion["_id"]
                x1 = x1_data[suggestion_id % len(x1_data)]
                outcome = {
                    "_id": suggestion["_id"],
                    "objective1": 0.1,
                    "objective2": 0.2,
                }
                outcomes.append(outcome)
            return outcomes
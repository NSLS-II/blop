# Conversion from databroker to tiled
This describes the process taken to upgrade blop to use tiled and how to use it to run the various tutorials and basic testing


## Starting the server
To start up the server open a new terminal and, in your enviornment do the command `tiled serve config config_tiled.yml` The server runs on **localhost:8000** and if a different port is needed, it can be changed on [src/blop/utils/prepare_re_env.py] (prepare_re_env.py) under `SERVER_HOST_LOCATION` 

### Tutorials
When `%run -i $prepare_re_env.__file__ --db-type=temp` is ran, the command palette is opened, asking for an authentication provider. Currently only option 2 is supported. Than for both username and password *bill* should be used. If login is successful, "you have logged in" should appear.

After this set up, no other work has to be done.

### Command Line Testing
In order to do simple testing, you can run the server in the command line via a different terminal than from where the server was started. 

To manually try out Tiled, you have to run the following in ipython. Where you can change the localhost port to something different if desired.
```.py
from tiled.client import from_uri
client = from_uri("http://localhost:8000")
client.login()
```
As above, you should select the authentication provider 2 with both the username and password being *bill*. 

If you want to manually add data, you can do that via
```
client['local']['raw'].write_array([1,2,3])
```
which will create the array `[1,2,3]` with a random uuid. If you would like to add a specific key to a array, you can do that with the `key` parameter as shown below
```
client['local']['raw'].write_array([1,2,3], key = 'array_key_name')
```
If you would like to view an array and you know the specific key, you can do 
```
client['local']['raw']['test_write'].read()
```
If you do not know the key, or would like a list of the keys being stored, you can run
```
list(client['local']['raw'])
```

## Debugging and viewing data
To see where the data is stored go to the file `file:///tmp/sirepo-bluesky-data/data` located in the config_tiled.yml under writable_storage. 

The above file hold the data that the server currently holds. It currently only able to the data that has been directly added to the database via .write_array() function. This information is stored based on the keys (either the uuid or custom name). By clicking into the key you want, you can see the shape of the array stored at that specific location

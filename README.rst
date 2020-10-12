Audio Perception
================


Dependencies
------------

- `NumPy <http://www.numpy.org/>`_
- `PyTorch <http://pytorch.org/>`_
- `Gnuplot <http://www.gnuplot.info/>`_
- `gnuplot-py <https://pypi.python.org/pypi/gnuplot-py/>`_
- `perception_msgs <https://github.com/idiap/perception_msgs/>`_
- `apkit <https://github.com/idiap/apkit/>`_
- `archs <https://github.com/idiap/archs/>`_

Sound Source Localization
-------------------------

**Script**

- ``ssl_nn.py``

**Subscribe topics**

- ``/naoqi_driver_node/audio`` : Audio signals captured by Pepper

**Publish topics**

- ``/audio_perception/ssl`` : sound detected
- ``/visualization_marker`` : markers of detected sound, which can be visualized in rviz

**Example Usage**

::

  rosrun audio_perception ssl_nn.py -n stft --context-frames=25 -v --marker --frame-id=Head PATH

A trained model (consists of two files) can be downloaded from http://protolab.aldebaran.com/mummer_downloads/idiap/models/

See also the example launch file ``launch/ssl_example.launch``.

**Arguments**

- ``PATH`` : the path to the trained neural network model without its extension

**Options**

- ``--cpu``        : use CPU for computation (instead of GPU)
- ``--visualize``  : visualize localization 
- ``--marker``     : publish ``/visualization_marker``
- ``--frame-id``   : specify the reference frame of detected sound location

For more detailed help message, run ``rosrun audio_perception ssl_nn.py -h``.

**Visualization**

There are two ways to visualize the SSL results.

1. Visualize the likelihood of sound source being in each individual direction in a top view (with option ``--visualize``).
#. Publish detection results as visulization markers (with option ``--marker``). You can view them in *rviz* by adding topic ``/visualization_marker`` to visualization. To overlay the markers to the pepper RGB camera images, you can view the topic ``/naoqi_driver_node/camera/front`` as *camera*. A correct reference frame of the markers must be specifed with option ``--frame-id``.


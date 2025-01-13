# mmDrone-Surv Project
## Drone detection and tracking with camera and mmWave radar
### What it's about?
This is a project about drone detection and tracking with camera and mmWave radar.
The main idea is, the camera pointed at the sky would identify a moving object, calculate the direction of the object,
send the directional information to the radar, then the radar would steer its beam forming sensors to that direction to
identify the radar.

### Why two modes of detection?
With camera, it's easier to identify drones near distance. For an object very far, it's very hard to identify any object
considering the sizes of drones nowadays can become very small. On the other hand, with the help of beam-forming, a
radar can project its signal to a very far distance and can detect drones, because of their unique signatures. The
drawback is that, the beamforming direction has to be specific, and it takes a lot of power and resources to point to
that direction. So, our main goal is to detect drones via beamforming techniques through mmWave radar. But to do that,
we need the specific direction that the radar need to point to. That comes from the camera, as small moving object can 
be identified via camera easily.

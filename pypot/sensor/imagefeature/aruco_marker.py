from multiprocessing import Process, Queue

from cv2 import aruco
import numpy as np

from ...robot.controller import SensorsController
from ...robot.sensor import Sensor


class ArucoMarker(Sensor):
    registers = Sensor.registers + ['position', 'id']

    def __init__(self, corners, id, camera_matrix, distortion ,size=0.018):
        """"
            Class to detect aruco markers, using like markers
            @size: real size in meters of marker, it's to important to get a better accuracy in
                getPositionMarker.
            @camera_matrix: matrix with intrinsec parameters of camera.
            @distortion: distortion of camera.
        """
        Sensor.__init__(self, 'aruco_marker_{}'.format(id))
        self.size = size
        self.camera_matrix = np.float64(camera_matrix)
        self.distortion = np.float64(distortion)
        self.position = aruco.estimatePoseSingleMarkers(corners,self.size,self.camera_matrix,self.distortion)
        self.id = id


    def __getattr__(self, attr):
        return getattr(self, attr)

    @property
    def json(self):
        return {"id": self.id, "position": self.position}


class ArucoMarkerDetector(SensorsController):
    def __init__(self, robot, name, cameras, intrinsec, distortion, freq, multiprocess=True, dictionary=aruco.DICT_6X6_250):
        SensorsController.__init__(self, None, [], freq)

        self.name = name
        self.dictionary = aruco.getPredefinedDictionary(dictionary)
        self._robot = robot
        self._names = cameras
        self.intrinsec = intrinsec
        self.distortion = distortion
        self.detect = (lambda img: self._bg_detection(img)
                       if multiprocess else aruco.detectMarkers(img,self.dictionary)[0:2])

    def update(self):
        if not hasattr(self, 'cameras'):
            self.cameras = [getattr(self._robot, c) for c in self._names]

        self._markers = sum([self.detect(c.frame) for c in self.cameras], [])
        self.sensors = [ArucoMarker(self._markers[0][i],self._markers[1][i],self.intrinsec,self.distortion) for i in range(0,len(self._markers[0]))]
            
    @property
    def markers(self):
        return self.sensors

    @property
    def registers(self):
        return ['aruco_markers']

    def _detect(self, q, img):
        """
            Detect the aruco marker and return [corners,ids] in format:

                [
                    [corners_tag1,corners_tag2,...,corners_tagn],
                    array([id_tag1, id_tag2,..., id_tagn])
                ]
        """
	markers_detected = aruco.detectMarkers(img,self.dictionary)[0:2]
	q.put(list(markers_detected))
    def _bg_detection(self, img):
        if not hasattr(self, 'q'):
            self.q = Queue()

        p = Process(target=self._detect, args=(self.q, img))
        p.start()
        return self.q.get()

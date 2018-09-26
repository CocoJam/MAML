import numpy as np
from tensorflow.python.platform import flags

Flags = flags.FLAGS


class sinGen:
    def __init__(self, innerGradUpdate, metaUpdate, config={}):
        self.innerGradUpdate = innerGradUpdate
        self.metaUpdate = metaUpdate
        self.classNum = 1
        if(Flags.datasource == "sinusoid"):
            self.ampli = config.get("amp", [0.1, 5.0])
            self.phase = config.get("pahse", [0, np.pi])
            self.input_range = config.get("input_range", [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
            self.generate = self.generateSinWave

    def generateSinWave(self, train = True):
        amplitude = np.random.uniform(self.ampli[0],self.ampli[1], [self.metaUpdate])
        phase = np.random.uniform(self.phase[0],self.phase[1], [self.metaUpdate])
        outputs = np.zeros([self.metaUpdate, self.innerGradUpdate, self.dim_output])
        init_inputs = np.zeros([self.metaUpdate, self.innerGradUpdate, self.dim_input])
        for i in range (self.metaUpdate):
            init_inputs[i] = np.random.uniform(self.input_range[0],self.input_range[1], [self.classNum , 1])
            outputs[i] = amplitude[i] * np.sin(init_inputs[i]-phase[i])
        return init_inputs, outputs, amplitude , phase



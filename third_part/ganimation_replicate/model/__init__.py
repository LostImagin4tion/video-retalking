from .base_model import BaseModel
from .ganimation import GANimationModel
from .stargan import StarGANModel



def create_model(opt):
    # specify model name here
    if opt.model == "ganimation":
        instance = GANimationModel()
        instance.initialize('cuda:0')
    elif opt.model == "stargan":
        instance = StarGANModel()
        instance.initialize(opt)
    else:
        instance = BaseModel()
        instance.initialize(opt)
    instance.setup()
    return instance


# from qdls.reg.register import registers
from ..register import registers


class Model:
    pass 

@registers.model.register("model1")
class Model1(Model):
    pass

@registers.model.register("model2")
class Model2(Model):
    pass

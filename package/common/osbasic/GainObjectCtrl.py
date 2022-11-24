import gc

class GainObjectCtrl :

    def __init__(self):
       pass

    @classmethod
    def getObjectsById(self, id_):
        for obj in gc.get_objects():
            if id(obj) == id_:
                return obj
        raise Exception("No found")
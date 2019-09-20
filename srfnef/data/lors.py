from srfnef import nef_class, Any
from srfnef.ops.common.property_mixins import ShapePropertyMixin, LengthPropertyMixin
from srfnef.ops.common.magic_method_mixins import GetItemMixin


@nef_class
class Lors(ShapePropertyMixin,
           LengthPropertyMixin,
           GetItemMixin):
    data: Any

    @property
    def fst(self):
        return self.data[:, :3]

    @property
    def snd(self):
        return self.data[:, 3:6]

    @property
    def tof_values(self):
        return self.data[:, 6]

    def append(self, lors: 'Lors') -> 'Lors':
        import numpy as np
        return self.update(data = np.vstack((self.data, lors.data)))

from typing import TYPE_CHECKING

import numpy as np

from . import _lnumba as nb
# ======================================================================
f16 = np.float16
f32 = np.float32
f64 = np.float64

i8 = np.int8
i16 = np.int16
i32 = np.int32
i64 = np.int64

u8 = np.uint8
u16 = np.uint16
u32 = np.uint32
u64 = np.uint64
up = np.uintp

# ======================================================================
# TYPES
if TYPE_CHECKING:
    from typing import TypeAlias
    from typing import TypeVar
    from typing import TypeVarTuple

    T1 = TypeVar('T1', bound = int)
    T2 = TypeVar('T2', bound = int)
    T3 = TypeVar('T3', bound = int)
    T4 = TypeVar('T4', bound = int)
    T5 = TypeVar('T5', bound = int)
    T6 = TypeVar('T6', bound = int)
    T7 = TypeVar('T7', bound = int)
    T8 = TypeVar('T8', bound = int)

    Shape: TypeAlias = tuple[T1, ...]
    Shape0: TypeAlias = tuple[()]
    Shape1: TypeAlias = tuple[T1]
    Shape2: TypeAlias = tuple[T1, T2]
    Shape3: TypeAlias = tuple[T1, T2, T3]
    Shape4: TypeAlias = tuple[T1, T2, T3, T4]
    Shape5: TypeAlias = tuple[T1, T2, T3, T4, T5]
    Shape6: TypeAlias = tuple[T1, T2, T3, T4, T5, T6]
    Shape7: TypeAlias = tuple[T1, T2, T3, T4, T5, T6, T7]

    _Scalar = TypeVar('_Scalar', bound = np.generic, covariant = True)

    ShapeType = TypeVar('ShapeType', bound = Shape)
    ShapeTuple = TypeVarTuple('ShapeTuple')

    Array: TypeAlias = np.ndarray[ShapeType, np.dtype[_Scalar]]
    Array0: TypeAlias = Array[Shape0, _Scalar]
    Array1: TypeAlias = Array[tuple[int], _Scalar]
    Array2: TypeAlias = Array[tuple[int, int], _Scalar]
    Array3: TypeAlias = Array[tuple[int, int, int], _Scalar]
    Array4: TypeAlias = Array[tuple[int, int, int, int], _Scalar]
    Array5: TypeAlias = Array[tuple[int, int, int, int, int], _Scalar]
    Array6: TypeAlias = Array[tuple[int, int, int, int, int, int], _Scalar]
    Array7: TypeAlias = Array[tuple[int, int, int, int, int, int, int], _Scalar]

    Number: TypeAlias = int | float

    _Array: TypeAlias = Array[tuple[*ShapeTuple], _Scalar]

    BoolArray: TypeAlias = _Array[*ShapeTuple, np.bool_]

    # C64Array: TypeAlias = _Array[*ShapeTuple, c64]
    # C128Array: TypeAlias = _Array[*ShapeTuple, c128]
    # C256Array: TypeAlias = _Array[*ShapeTuple, c256]

    F16Array: TypeAlias = _Array[*ShapeTuple, f16]
    F32Array: TypeAlias = _Array[*ShapeTuple, f32]
    F64Array: TypeAlias = _Array[*ShapeTuple, f64]
    # F128Array: TypeAlias = _Array[*ShapeTuple, f128]
    FArray: TypeAlias = F16Array | F32Array | F64Array

    Floaty: TypeAlias = float | FArray

    I16Array: TypeAlias = _Array[*ShapeTuple, i16]
    I32Array: TypeAlias = _Array[*ShapeTuple, i32]
    I64Array: TypeAlias = _Array[*ShapeTuple, i64]
    IArray: TypeAlias = I64Array | I32Array | I64Array

    U16Array: TypeAlias = _Array[*ShapeTuple, u16]
    U32Array: TypeAlias = _Array[*ShapeTuple, u32]
    U64Array: TypeAlias = _Array[*ShapeTuple, u64]
    UPArray: TypeAlias = _Array[*ShapeTuple, up]
    UArray: TypeAlias = U16Array | U32Array | U64Array

    NumArray: TypeAlias = FArray | IArray | UArray

    Num: TypeAlias = NumArray | Number

    ScalarVar = TypeVar('ScalarVar', bound = np.generic)

    Start: TypeAlias = int
    Stop: TypeAlias = int
    Step: TypeAlias = int
    Length: TypeAlias = int
else:
    Callable = tuple

    Any = object

    Shape = Shape0 = Shape1 = Shape2 = Shape3 = Shape4 = Shape5 = type
    Shape6 = Shape7 = type
    Array = Array1 = Array2 = Array3 = Array4 = Array5 = Array6 = Array7 = tuple

    F16Array = F32Array = F64Array = FArray = Floaty = type
    I16Array = I32Array = I64Array = IArray = type
    U16Array = U32Array = U64Array = UArray = type
    NumArray = Num =  BoolArray = type
    ScalarVar = object
    Start = Stop = Step = Length = int
# ======================================================================
@nb.njit
def rfrange(n: int, dtype: type = f64):
    array = np.ones((n,), dtype)
    for i in range(2, n):
        array[i] = array[i-1] / i
    return array
# ======================================================================
@nb.njit
def rfrange_allocated(n: int,
                      dtype: ScalarVar = f64,
                      out: Array1[ScalarVar] = None):
    if out is None:
        array = np.ones((n,), dtype)
    for i in range(2, n):
        array[i] = array[i-1] / i
    return array
# ======================================================================
@nb.njit
def erange(value: float, n: int, dtype: type = f64):
    array = np.ones((n,), dtype)
    array[0] = value
    for i in range(1, n):
        array[i] = array[i-1] * value
    return array

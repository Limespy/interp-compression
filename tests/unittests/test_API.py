# import limesqueezer as ls
# import numpy as np
# import pytest
# # ======================================================================
# tol = (1e-3, 1e-4, 1)
# X_DATA = np.linspace(0, 6, int(1e4))
# Y_DATA1 = np.array(np.sin(X_DATA * X_DATA))
# Y_DATA2 = np.array((Y_DATA1, Y_DATA1[::-1])).T
# # ======================================================================
# # AUXILIARIES
# def f2zero_100(n: int) -> tuple[float, bool]:
#     """Returns < 0 for values 0 to 100 and >0 for values > 100."""
#     if round(n) != n: raise ValueError('Not whole number')
#     if n < 0: raise ValueError('Input must be >= 0')
#     return np.sqrt(n) - 10.01, True
# # ======================================================================
# def compressionaxis(x: ls.Float64Array, y: ls.Float64Array) -> int:
#     if y.ndim == 1:
#         return 0
#     if y.shape[0] == len(x):
#         return 0
#     if y.shape[1] == len(x):
#         return 1
#     raise ValueError(f'Unable to find axis on shapes {x.shape} and {y.shape}')
# # ======================================================================
# def assert_np_equal(left: ls.Float64Array, right: ls.Float64Array) -> None:
#     """Asserts that two Numpy arrays have same shape and have equal
#     elements."""
#     assert left.shape == right.shape
#     assert np.all(left == right)
# # ----------------------------------------------------------------------
# def assert_endpoint_equal(left: ls.Float64Array, right: ls.Float64Array
#                           ) -> None:
#     """Assers that two sequences have both first and last elements equal."""
#     assert_np_equal(left[0], right[0])
#     assert_np_equal(left[-1], right[-1])
# # ======================================================================
# # Block Compresion
# @pytest.mark.parametrize('y_data', (Y_DATA1,
#                                     ls.to_ndarray(Y_DATA1, (-1,1)),
#                                     Y_DATA2,
#                                     Y_DATA2.T))
# def test_compress_default(y_data):
#     xc, yc = ls.compress(X_DATA, y_data, tolerances = tol, keepshape = True)
#     assert y_data.ndim == yc.ndim
#     assert_endpoint_equal(X_DATA, xc)
#     # if compressionaxis(X_DATA, y_data):
#     #     assert_endpoint_equal(y_data.T, yc.T)
# # ----------------------------------------------------------------------
# @pytest.mark.parametrize('tolerances', ((1e-2, 1e-2, 0),
#                                         (1e-2, 1e-2),
#                                         (1e-2,),
#                                         1e-2))
# def test_block_tolerances_correct(tolerances):
#     """- Compression accepts different tolerance inputs
#     """
#     ls.compress(X_DATA, Y_DATA1, tolerances = tolerances)
# # ----------------------------------------------------------------------
# def test_block_tolerances_type_error():
#     """- Compression rejects incorrect tolerance inputs
#     """
#     with pytest.raises(TypeError):
#         ls.compress(X_DATA, Y_DATA1, tolerances = 'hmm')
# # ----------------------------------------------------------------------
# @pytest.mark.parametrize(('tolerances', ), (((),),
#                                             ((1,2,3,4),)))
# def test_block_tolerances_value_error(tolerances):
#     """- Compression rejects incorrect tolerance inputs
#     """
#     with pytest.raises(ValueError):
#         ls.compress(X_DATA, Y_DATA1, tolerances = tolerances)
# # ----------------------------------------------------------------------
# def test_block_2_3_tolerances_limits():
#     """- Compression works as expected at the edges of the tolerance
#     range
#     """
#     x_c, y_c = ls.compress(X_DATA, Y_DATA1,
#                             tolerances = np.finfo(float).max / 1e2,
#                             keepshape = True)
#     assert (2,) == x_c.shape
#     assert (2,) == y_c.shape
#     x_c, y_c = ls.compress(X_DATA, Y_DATA1,
#                             tolerances = np.finfo(float).eps,
#                             keepshape = True)
#     assert X_DATA.shape == x_c.shape
#     assert Y_DATA1.shape == y_c.shape
# # ----------------------------------------------------------------------
# def test_block_3_1_keepshape():
#     """- Array noncompressed dimension is kept same"""
#     x_c, y_c = ls.compress(X_DATA, Y_DATA2, keepshape = True)
#     assert len(X_DATA.shape) == len(x_c.shape)
#     assert Y_DATA2.shape[1] == y_c.shape[1]
# # ======================================================================
# # Stream Compression
# def test_stream_1_1y():
#     """Stream compression runs and outputs correctly."""
#     # ------------------------------------------------------------------
#     with ls.Stream(X_DATA[0], Y_DATA1[0], tolerances = tol) as record:
#         assert (isinstance(record, ls.StreamRecord))
#         assert record.state == 'open'
#         for x, y in zip(X_DATA[1:], Y_DATA1[1:]):
#             record(x, y)
#     # ------------------------------------------------------------------
#     assert record.state == 'closed'
#     assert_endpoint_equal(X_DATA, record.x)
#     assert_endpoint_equal(ls.to_ndarray(Y_DATA1, (-1, 1)), record.y)
# # ======================================================================
# # Stream Compression
# def test_stream_vs_block_3_1y():
#     """Block and stream compressions must give equal compressed output for 1 y
#     variable."""
#     xc_block, yc_block = ls.compress(X_DATA, Y_DATA1, tolerances = tol,
#                             initial_step = 100, errorfunction = 'MaxAbs')
#     # ------------------------------------------------------------------
#     with ls.Stream(X_DATA[0], Y_DATA1[0], tolerances = tol) as record:
#         for x, y in zip(X_DATA[1:], Y_DATA1[1:]):
#             record(x, y)
#     # ------------------------------------------------------------------
#     assert_np_equal(xc_block, record.x)
#     assert_np_equal(yc_block, record.y)
# # ======================================================================
# # Stream Compression
# def test_stream_vs_block_3_1y_numba():
#     """Block and stream compressions must give equal compressed output for 1 y
#     variable using Numba."""
#     xc_block, yc_block = ls.compress(X_DATA, Y_DATA1, tolerances = tol,
#                                     initial_step = 100,
#                                     errorfunction = 'MaxAbs',
#                                     use_numba = 1)
#     # ------------------------------------------------------------------
#     with ls.Stream(X_DATA[0], Y_DATA1[0], tolerances = tol, use_numba = 1
#                     ) as record:
#         for x, y in zip(X_DATA[1:], Y_DATA1[1:]):
#             record(x, y)
#     # ------------------------------------------------------------------
#     assert_np_equal(xc_block, record.x)
#     assert_np_equal(yc_block, record.y)
# # ======================================================================
# def test_stream_vs_block_3_2y():
#     """Block and stream compressions must give equal compressed output for 1 y
#     variable."""
#     xc_block, yc_block = ls.compress(X_DATA, Y_DATA2, tolerances = tol,
#                             initial_step = 100, errorfunction = 'MaxAbs')
#     # ------------------------------------------------------------------
#     with ls.Stream(X_DATA[0], Y_DATA2[0], tolerances = tol) as record:
#         for x, y in zip(X_DATA[1:], Y_DATA2[1:]):
#             record(x, y)
#     # ------------------------------------------------------------------
#     assert_np_equal(xc_block, record.x)
#     assert_np_equal(yc_block, record.y)
# # ======================================================================
# def test_stream_vs_block_3_2y_numba():
#     """Block and stream compressions must give equal compressed output for 1 y
#     variable using Numba."""
#     xc_block, yc_block = ls.compress(X_DATA, Y_DATA2, tolerances = tol,
#                                     initial_step = 100,
#                                     errorfunction = 'MaxAbs',
#                                     use_numba = 1)
#     # ------------------------------------------------------------------
#     with ls.Stream(X_DATA[0], Y_DATA2[0], tolerances = tol, use_numba = 1
#                     ) as record:
#         for x, y in zip(X_DATA[1:], Y_DATA2[1:]):
#             record(x, y)
#     # ------------------------------------------------------------------
#     assert_np_equal(xc_block, record.x)
#     assert_np_equal(yc_block, record.y)
# # ======================================================================
# # Decompression
# def test_decompress_1_mock():
#     """Runs decompression on and compares to original."""
#     assert (np.allclose(ls.decompress(X_DATA, Y_DATA1)(X_DATA),
#                                 Y_DATA1, atol = 1e-14))

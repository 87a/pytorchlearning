要想通过Tensor类本身就支持了使用autograd功能，只需要设置.requires_grad=True

```python
x = torch.rand(5, 5, requires_grad=True)
print(x)

y = torch.rand(5, 5, requires_grad=True)
print(y)

z = torch.sum(x + y)
print(z)
```
输出
```
tensor([[0.3045, 0.9751, 0.0611, 0.7479, 0.3092],
        [0.4586, 0.1826, 0.4967, 0.9269, 0.4115],
        [0.0776, 0.8710, 0.6885, 0.2775, 0.0659],
        [0.0723, 0.0941, 0.8108, 0.3236, 0.6857],
        [0.5748, 0.3782, 0.7934, 0.4094, 0.9056]], requires_grad=True)
tensor([[0.4793, 0.4947, 0.9852, 0.4600, 0.3822],
        [0.2572, 0.1336, 0.1529, 0.9275, 0.0515],
        [0.7044, 0.2764, 0.1592, 0.6278, 0.6247],
        [0.7006, 0.3895, 0.7969, 0.8764, 0.3585],
        [0.0664, 0.1253, 0.4861, 0.4079, 0.9335]], requires_grad=True)
tensor(23.7604, grad_fn=<SumBackward0>)
```

# 简单的自动求导
```python
z.backward()
print(x.grad, '\n', y.grad)
```
输出
```
tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]]) 
 tensor([[1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]])
```

如果Tensor类表示的是一个**标量**（即它包含一个元素的张量），则不需要为`backward()`指定任何参数，但是如果它有更多的元素，则需要指定一个`gradient`参数，它是形状匹配的张量。 以上的 z.`backward()`相当于是`z.backward(torch.tensor(1.))`的简写。 这种参数常出现在图像分类中的单标签分类，输出一个标量代表图像的标签。

# 复杂的自动求导
```python
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = x ** 2 + y ** 3
print(z)

#我们的返回值不是一个标量，所以需要输入一个大小相同的张量作为参数，这里我们用ones_like函数根据x生成一个张量
z.backward(torch.ones_like(x))
print(x.grad)
```
输出
```
tensor([[1.5054, 0.6936, 0.0346, 0.0306, 0.7420],
        [0.4883, 0.6614, 0.1533, 0.0444, 0.1823],
        [1.0614, 0.8009, 0.6204, 0.8760, 0.5847],
        [0.0791, 1.3400, 0.5347, 0.7434, 0.5843],
        [0.7194, 0.3020, 0.0374, 0.7661, 1.4984]], grad_fn=<AddBackward0>)
tensor([[1.9787, 1.6460, 0.3109, 0.2735, 1.3470],
        [1.3967, 0.9447, 0.7822, 0.4213, 0.7584],
        [0.8550, 1.5597, 1.5363, 1.5817, 0.7570],
        [0.1642, 1.1694, 1.2445, 1.5290, 1.5287],
        [1.6964, 0.1752, 0.0147, 1.0448, 1.5135]])
```

我们可以使用`with torch.no_grad()`上下文管理器临时禁止对已设置`requires_grad=True`的张量进行自动求导。这个方法在测试集计算准确率的时候会经常用到，例如
```python
with torch.no_grad():
    print((x + y * 2).requires_grad)
```
输出
```
False
```

使用`.no_grad()`进行嵌套后，代码不会跟踪历史记录，也就是说保存的这部分记录会减少内存的使用量并且会加快少许的运算速度

# Autograd 过程解析
为了说明Pytorch的自动求导原理，我们来尝试分析一下PyTorch的源代码，虽然Pytorch的 Tensor和 TensorBase都是使用CPP来实现的，但是可以使用一些Python的一些方法查看这些对象在Python的属性和状态。 Python的 `dir() `返回参数的属性、方法列表。z是一个Tensor变量，看看里面有哪些成员变量。
```python
print(dir(z))
```
输出
```
['T', '__abs__', '__add__', '__and__', '__array__', '__array_priority__', '__array_wrap__', '__bool__', '__class__', '__complex__', '__contains__', '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__div__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__ilshift__', '__imod__', '__imul__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__long__', '__lshift__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__new__', '__nonzero__', '__or__', '__pos__', '__pow__', '__radd__', '__rdiv__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rmul__', '__rpow__', '__rshift__', '__rsub__', '__rtruediv__', '__setattr__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__torch_function__', '__truediv__', '__weakref__', '__xor__', '_backward_hooks', '_base', '_cdata', '_coalesced_', '_dimI', '_dimV', '_grad', '_grad_fn', '_indices', '_is_view', '_make_subclass', '_nnz', '_reduce_ex_internal', '_to_sparse_csr', '_update_names', '_values', '_version', 'abs', 'abs_', 'absolute', 'absolute_', 'acos', 'acos_', 'acosh', 'acosh_', 'add', 'add_', 'addbmm', 'addbmm_', 'addcdiv', 'addcdiv_', 'addcmul', 'addcmul_', 'addmm', 'addmm_', 'addmv', 'addmv_', 'addr', 'addr_', 'align_as', 'align_to', 'all', 'allclose', 'amax', 'amin', 'angle', 'any', 'apply_', 'arccos', 'arccos_', 'arccosh', 'arccosh_', 'arcsin', 'arcsin_', 'arcsinh', 'arcsinh_', 'arctan', 'arctan_', 'arctanh', 'arctanh_', 'argmax', 'argmin', 'argsort', 'as_strided', 'as_strided_', 'as_subclass', 'asin', 'asin_', 'asinh', 'asinh_', 'atan', 'atan2', 'atan2_', 'atan_', 'atanh', 'atanh_', 'backward', 'baddbmm', 'baddbmm_', 'bernoulli', 'bernoulli_', 'bfloat16', 'bincount', 'bitwise_and', 'bitwise_and_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_xor', 'bitwise_xor_', 'bmm', 'bool', 'broadcast_to', 'byte', 'cauchy_', 'cdouble', 'ceil', 'ceil_', 'cfloat', 'char', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 'chunk', 'clamp', 'clamp_', 'clamp_max', 'clamp_max_', 'clamp_min', 'clamp_min_', 'clip', 'clip_', 'clone', 'coalesce', 'col_indices', 'conj', 'contiguous', 'copy_', 'copysign', 'copysign_', 'cos', 'cos_', 'cosh', 'cosh_', 'count_nonzero', 'cpu', 'cross', 'crow_indices', 'cuda', 'cummax', 'cummin', 'cumprod', 'cumprod_', 'cumsum', 'cumsum_', 'data', 'data_ptr', 'deg2rad', 'deg2rad_', 'dense_dim', 'dequantize', 'det', 'detach', 'detach_', 'device', 'diag', 'diag_embed', 'diagflat', 'diagonal', 'diff', 'digamma', 'digamma_', 'dim', 'dist', 'div', 'div_', 'divide', 'divide_', 'dot', 'double', 'dsplit', 'dtype', 'eig', 'element_size', 'eq', 'eq_', 'equal', 'erf', 'erf_', 'erfc', 'erfc_', 'erfinv', 'erfinv_', 'exp', 'exp2', 'exp2_', 'exp_', 'expand', 'expand_as', 'expm1', 'expm1_', 'exponential_', 'fill_', 'fill_diagonal_', 'fix', 'fix_', 'flatten', 'flip', 'fliplr', 'flipud', 'float', 'float_power', 'float_power_', 'floor', 'floor_', 'floor_divide', 'floor_divide_', 'fmax', 'fmin', 'fmod', 'fmod_', 'frac', 'frac_', 'frexp', 'gather', 'gcd', 'gcd_', 'ge', 'ge_', 'geometric_', 'geqrf', 'ger', 'get_device', 'grad', 'grad_fn', 'greater', 'greater_', 'greater_equal', 'greater_equal_', 'gt', 'gt_', 'half', 'hardshrink', 'has_names', 'heaviside', 'heaviside_', 'histc', 'hsplit', 'hypot', 'hypot_', 'i0', 'i0_', 'igamma', 'igamma_', 'igammac', 'igammac_', 'imag', 'index_add', 'index_add_', 'index_copy', 'index_copy_', 'index_fill', 'index_fill_', 'index_put', 'index_put_', 'index_select', 'indices', 'inner', 'int', 'int_repr', 'inverse', 'is_coalesced', 'is_complex', 'is_contiguous', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_leaf', 'is_meta', 'is_mkldnn', 'is_mlc', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xpu', 'isclose', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'isreal', 'istft', 'item', 'kron', 'kthvalue', 'layout', 'lcm', 'lcm_', 'ldexp', 'ldexp_', 'le', 'le_', 'lerp', 'lerp_', 'less', 'less_', 'less_equal', 'less_equal_', 'lgamma', 'lgamma_', 'log', 'log10', 'log10_', 'log1p', 'log1p_', 'log2', 'log2_', 'log_', 'log_normal_', 'log_softmax', 'logaddexp', 'logaddexp2', 'logcumsumexp', 'logdet', 'logical_and', 'logical_and_', 'logical_not', 'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logit', 'logit_', 'logsumexp', 'long', 'lstsq', 'lt', 'lt_', 'lu', 'lu_solve', 'map2_', 'map_', 'masked_fill', 'masked_fill_', 'masked_scatter', 'masked_scatter_', 'masked_select', 'matmul', 'matrix_exp', 'matrix_power', 'max', 'maximum', 'mean', 'median', 'min', 'minimum', 'mm', 'mode', 'moveaxis', 'movedim', 'msort', 'mul', 'mul_', 'multinomial', 'multiply', 'multiply_', 'mv', 'mvlgamma', 'mvlgamma_', 'name', 'names', 'nan_to_num', 'nan_to_num_', 'nanmedian', 'nanquantile', 'nansum', 'narrow', 'narrow_copy', 'ndim', 'ndimension', 'ne', 'ne_', 'neg', 'neg_', 'negative', 'negative_', 'nelement', 'new', 'new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'nextafter', 'nextafter_', 'nonzero', 'norm', 'normal_', 'not_equal', 'not_equal_', 'numel', 'numpy', 'orgqr', 'ormqr', 'outer', 'output_nr', 'permute', 'pin_memory', 'pinverse', 'polygamma', 'polygamma_', 'positive', 'pow', 'pow_', 'prelu', 'prod', 'put', 'put_', 'q_per_channel_axis', 'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 'qr', 'qscheme', 'quantile', 'rad2deg', 'rad2deg_', 'random_', 'ravel', 'real', 'reciprocal', 'reciprocal_', 'record_stream', 'refine_names', 'register_hook', 'reinforce', 'relu', 'relu_', 'remainder', 'remainder_', 'rename', 'rename_', 'renorm', 'renorm_', 'repeat', 'repeat_interleave', 'requires_grad', 'requires_grad_', 'reshape', 'reshape_as', 'resize', 'resize_', 'resize_as', 'resize_as_', 'retain_grad', 'roll', 'rot90', 'round', 'round_', 'rsqrt', 'rsqrt_', 'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'select', 'set_', 'sgn', 'sgn_', 'shape', 'share_memory_', 'short', 'sigmoid', 'sigmoid_', 'sign', 'sign_', 'signbit', 'sin', 'sin_', 'sinc', 'sinc_', 'sinh', 'sinh_', 'size', 'slogdet', 'smm', 'softmax', 'solve', 'sort', 'sparse_dim', 'sparse_mask', 'sparse_resize_', 'sparse_resize_and_clear_', 'split', 'split_with_sizes', 'sqrt', 'sqrt_', 'square', 'square_', 'squeeze', 'squeeze_', 'sspaddmm', 'std', 'stft', 'storage', 'storage_offset', 'storage_type', 'stride', 'sub', 'sub_', 'subtract', 'subtract_', 'sum', 'sum_to_size', 'svd', 'swapaxes', 'swapaxes_', 'swapdims', 'swapdims_', 'symeig', 't', 't_', 'take', 'take_along_dim', 'tan', 'tan_', 'tanh', 'tanh_', 'tensor_split', 'tile', 'to', 'to_dense', 'to_mkldnn', 'to_sparse', 'tolist', 'topk', 'trace', 'transpose', 'transpose_', 'triangular_solve', 'tril', 'tril_', 'triu', 'triu_', 'true_divide', 'true_divide_', 'trunc', 'trunc_', 'type', 'type_as', 'unbind', 'unflatten', 'unfold', 'uniform_', 'unique', 'unique_consecutive', 'unsafe_chunk', 'unsafe_split', 'unsafe_split_with_sizes', 'unsqueeze', 'unsqueeze_', 'values', 'var', 'vdot', 'view', 'view_as', 'vsplit', 'where', 'xlogy', 'xlogy_', 'xpu', 'zero_']
```


返回很多，我们直接排除掉一些Python中特殊方法（以__开头和结束的）和私有方法（以_开头的，直接看几个比较主要的属性：

##  `.is_leaf`
记录是否是叶子节点。通过这个属性来确定这个变量的类型 在官方文档中所说的“graph leaves”，“leaf variables”，都是指像x，y这样的**手动创建的、而非运算得到**的变量，这些变量成为创建变量。 像z这样的，是通过计算后得到的结果称为结果变量。

一个变量是创建变量还是结果变量是通过`.is_leaf`来获取的。
```python
print("x.is_leaf="+str(x.is_leaf))
print("z.is_leaf="+str(z.is_leaf))
```
输出
```
x.is_leaf=True
z.is_leaf=False
```

x是手动创建的没有通过计算，所以他被认为是一个叶子节点也就是一个创建变量，而z是通过x与y的一系列计算得到的，所以不是叶子结点也就是结果变量。

为什么我们执行z.backward()方法会更新x.grad和y.grad呢？ .grad_fn属性记录的就是这部分的操作，虽然.backward()方法也是CPP实现的，但是可以通过Python来进行简单的探索。

## `grad_fn`
记录并且编码了完整的计算历史
```python
print(z.grad_fn)
```
输出
```
<AddBackward0 object at 0x0000020B5EE58700>
```
`grad_fn`是一个AddBackward0类型的变量 AddBackward0这个类也是用Cpp来写的，但是我们从名字里就能够大概知道，他是加法(ADD)的反反向传播（Backward），看看里面有些什么东西
```python
print(dir(z.grad_fn))
```
输出
```
['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_register_hook_dict', '_saved_alpha', 'metadata', 'name', 'next_functions', 'register_hook', 'requires_grad']
```

next_functions就是grad_fn的精华
## next_functions
```python
print(z.grad_fn.next_functions)
```
输出
```
((<PowBackward0 object at 0x0000028D33558700>, 0), (<PowBackward0 object at 0x0000028D335581C0>, 0))
```

next_functions是一个tuple of tuple of PowBackward0 and int。

为什么是2个tuple ？ 因为我们的操作是$z=x^2+y^3$刚才的AddBackward0是相加，而前面的操作是乘方 `PowBackward0`。tuple第一个元素就是x相关的操作记录.

```python
xg = z.grad_fn.next_functions[0][0]
print(dir(xg))
```
输出
```
['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '_register_hook_dict', '_saved_exponent', '_saved_self', 'metadata', 'name', 'next_functions', 'register_hook', 'requires_grad']
```


继续深挖
```python
x_leaf = xg.next_functions[0][0]
print(type(x_leaf))
```
输出
```
<class 'AccumulateGrad'>
```
## `.variable`
在PyTorch的反向图计算中，`AccumulateGrad`类型代表的就是叶子节点类型，也就是计算图终止节点。`AccumulateGrad`类中有一个`.variable`属性指向叶子节点.

```python
print(x_leaf.variable)
```
输出
```
tensor([[0.3178, 0.9205, 0.0472, 0.6913, 0.3198],
        [0.4680, 0.1733, 0.7476, 0.3120, 0.3863],
        [0.8383, 0.9174, 0.3818, 0.1291, 0.2249],
        [0.5527, 0.2561, 0.4008, 0.7109, 0.5796],
        [0.4463, 0.5087, 0.4701, 0.4638, 0.7504]], requires_grad=True)
```

这个.variable的属性就是我们的生成的变量x
```python
print("x_leaf.variable的id:"+str(id(x_leaf.variable)))
print("x的id:"+str(id(x)))
```
输出
```
x_leaf.variable的id:2739664700800
x的id:2739664700800
```

## 总结
这样整个规程就很清晰了：

1. 当我们执行`z.backward()`的时候。这个操作将调用z里面的`grad_fn`这个属性，执行求导的操作。
2. 这个操作将遍历`grad_fn`的`next_functions`，然后分别取出里面的Function（AccumulateGrad），执行求导操作。这部分是一个递归的过程直到最后类型为叶子节点。
3. 计算出结果以后，将结果保存到他们对应的`variable` 这个变量所引用的对象（x和y）的 `grad`这个属性里面。
4. 求导结束。所有的叶节点的`grad`变量都得到了相应的更新


最终当我们执行完`z.backward()`之后，a和b里面的`grad`值就得到了更新。

# 扩展Autograd
如果需要自定义autograd扩展新的功能，就需要**扩展Function类**。因为Function使用autograd来计算结果和梯度，并对操作历史进行编码。 在Function类中最主要的方法就是`forward()`和`backward()`他们分别代表了前向传播和反向传播。

一个自定义的Function需要一下三个方法：

`__init__ (optional)`：如果这个操作需要额外的参数则需要定义这个Function的构造函数，不需要的话可以忽略。

`forward()`：执行前向传播的计算代码

`backward()`：反向传播时梯度计算的代码。 参数的个数和`forward`返回值的个数一样，每个参数代表传回到此操作的梯度。

```python
from torch.autograd.function import Function


class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx 用来保存信息这里类似self，并且ctx的属性可以在backward中调用
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # 返回的参数要与输入的参数一样.
        # 第一个输入为3x3的张量，第二个为一个常数
        # 常数的梯度必须是 None.
        return grad_output * ctx.constant, None


a = torch.rand(3, 3, requires_grad=True)
b = MulConstant.apply(a, 5)
print("a:" + str(a))
print("b:" + str(b))  # b为a的元素乘以5

b.backward(torch.ones_like(a))
print(a.grad)
```

输出
```
a:tensor([[0.1551, 0.6563, 0.8274],
        [0.0635, 0.5604, 0.8485],
        [0.4865, 0.8582, 0.8889]], requires_grad=True)
b:tensor([[0.7755, 3.2815, 4.1370],
        [0.3177, 2.8020, 4.2425],
        [2.4324, 4.2910, 4.4446]], grad_fn=<MulConstantBackward>)
tensor([[5., 5., 5.],
        [5., 5., 5.],
        [5., 5., 5.]])
```


import neuronxcc.starfish.penguin.ir.ir as m0
import neuronxcc.starfish.penguin.ir.DebugInfo as m1
import neuronxcc.starfish.penguin.targets.tonga.APIndex as m2
import neuronxcc.starfish.penguin.targets.tonga.TongaInst as m3
import neuronxcc.starfish.penguin.targets.tonga.TongaISAInst as m4
import neuronxcc.starfish.penguin.targets.tonga.TongaTensor as m5
import numpy as np
v0 = m0.Function(id_=0, batch_ids=[], attrs=("model-type=memory-bound","mac-count=0",'hlo-metrics={"AliasedOutputSize":0,"ArithmeticIntensity":0.0,"ConstantSize":0,"HloInputCount":-1,"HloMacCount":0,"HloOutputCount":-1,"IfmapSize":0,"OfmapSize":0,"OutputsReadFromCount":-1,"PassthroughTensorsCount":-1,"RedundantOutputCount":-1,"Traffic":302518272}'))
def weight_load(p):
  t = np.load(p)
  return t
import neuronxcc.starfish.support as m7
v1 = m0.Tensor(name="input0", shape=(128,2048), parent=v0, id=1, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,2048), layout="NC", transpose=(0,1)), attrs={'CrossPassTensor': ""})
v0.markInput(v1)
v2 = m0.Tensor(name="input1", shape=(2048,), parent=v0, id=2, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(2048,), layout="N", transpose=(0,)), attrs={'CrossPassTensor': ""})
v0.markInput(v2)
v3 = m0.Tensor(name="input2", shape=(128,192,2048), parent=v0, id=3, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,192,2048), layout="NHC", transpose=(0,1,2)), attrs={'CrossPassTensor': ""})
v0.markInput(v3)
v4 = m0.Tensor(name="input3", shape=(128,2048,384), parent=v0, id=4, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,2048,384), layout="NHC", transpose=(0,1,2)), attrs={'CrossPassTensor': ""})
v0.markInput(v4)
v5 = m0.Tensor(name="output2", shape=(128,192,2048), parent=v0, id=5, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,192,2048), layout="NHC", transpose=(0,1,2)), )
v6 = m0.OffloadedMemCpy(srcs=[v3], dsts=[v5], dtype=m7.dtype.bfloat16, id=6, parent=v0, dl=m1.DebugLocation(tensor_op_name="UnnamedModule", file="", line=0, column=0, hlo_id=0))
v7 = m0.Tensor(name="reshape.1", shape=(128,16,128), parent=v0, id=7, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,16,128), layout="NHC", transpose=(0,1,2)), )
v8 = m0.OffloadedMemCpy(srcs=[v1], dsts=[v7], dtype=m7.dtype.bfloat16, id=8, parent=v0, dl=m1.DebugLocation(tensor_op_name="_reshape.6", file="", line=0, column=0, hlo_id=21))
v10 = m0.Tensor(name="transpose.1", shape=(128,16,128), parent=v0, id=9, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,16,128), layout="NHC", transpose=(0,1,2)), )
import neuronxcc.starfish.penguin.frontends.XlaFE as m8
v9 = m8.NeuronTensorOp(srcs=[v7], dsts=[v10], xla_op='mhlo.transpose', src_shape=(128,16,128), permutation=(2,1,0), id=10, parent=v0, dl=m1.DebugLocation(tensor_op_name="_transpose.3", file="", line=0, column=0, hlo_id=22))
v11 = m0.Tensor(name="output0", shape=(128,2048), parent=v0, id=11, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,2048), layout="NC", transpose=(0,1)), )
v12 = m0.OffloadedMemCpy(srcs=[v10], dsts=[v11], dtype=m7.dtype.bfloat16, id=12, parent=v0, dl=m1.DebugLocation(tensor_op_name="_reshape.7", file="", line=0, column=0, hlo_id=23))
v13 = m0.Tensor(name="reshape.2", shape=(16,128), parent=v0, id=13, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(16,128), layout="NC", transpose=(0,1)), )
v14 = m0.OffloadedMemCpy(srcs=[v2], dsts=[v13], dtype=m7.dtype.bfloat16, id=14, parent=v0, dl=m1.DebugLocation(tensor_op_name="_reshape.8", file="", line=0, column=0, hlo_id=24))
v16 = m0.Tensor(name="transpose.2", shape=(128,16), parent=v0, id=15, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,16), layout="NC", transpose=(0,1)), )
v15 = m8.NeuronTensorOp(srcs=[v13], dsts=[v16], xla_op='mhlo.transpose', src_shape=(16,128), permutation=(1,0), id=16, parent=v0, dl=m1.DebugLocation(tensor_op_name="_transpose.4", file="", line=0, column=0, hlo_id=25))
v17 = m0.Tensor(name="output1", shape=(2048,), parent=v0, id=17, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(2048,), layout="N", transpose=(0,)), )
v18 = m0.OffloadedMemCpy(srcs=[v16], dsts=[v17], dtype=m7.dtype.bfloat16, id=18, parent=v0, dl=m1.DebugLocation(tensor_op_name="_reshape.9", file="", line=0, column=0, hlo_id=26))
v19 = m0.Tensor(name="reshape.3", shape=(128,16,128,2,2,96), parent=v0, id=19, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,16,128,2,2,96), layout="", transpose=(0,1,2,3,4,5)), )
v20 = m0.OffloadedMemCpy(srcs=[v4], dsts=[v19], dtype=m7.dtype.bfloat16, id=20, parent=v0, dl=m1.DebugLocation(tensor_op_name="_reshape.10", file="", line=0, column=0, hlo_id=27))
v22 = m0.Tensor(name="transpose.3", shape=(128,2,128,16,2,96), parent=v0, id=21, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,2,128,16,2,96), layout="", transpose=(0,1,2,3,4,5)), )
v21 = m8.NeuronTensorOp(srcs=[v19], dsts=[v22], xla_op='mhlo.transpose', src_shape=(128,16,128,2,2,96), permutation=(0,4,2,1,3,5), id=22, parent=v0, dl=m1.DebugLocation(tensor_op_name="_transpose.5", file="", line=0, column=0, hlo_id=28))
v23 = m0.Tensor(name="output3", shape=(128,2048,384), parent=v0, id=23, dtype=m7.dtype.bfloat16, view=m0.TensorView(shape=(128,2048,384), layout="NHC", transpose=(0,1,2)), )
v24 = m0.OffloadedMemCpy(srcs=[v22], dsts=[v23], dtype=m7.dtype.bfloat16, id=24, parent=v0, dl=m1.DebugLocation(tensor_op_name="_reshape.11", file="", line=0, column=0, hlo_id=29))
v0.aliasTensors("output0", "input0", "must")
v0.aliasTensors("output1", "input1", "must")
v0.aliasTensors("output2", "input2", "must")
v0.aliasTensors("output3", "input3", "must")
v0.markOutput(v11)
v0.markOutput(v17)
v0.markOutput(v5)
v0.markOutput(v23)
v0.id=25
ir=v0

import numpy as np
from loguru import logger
from brocolli.converter.pytorch_graph import PytorchGraph
# import caffe.proto.caffe_pb2 as pb2
from brocolli.converter.ppl_caffe import caffe_pb2 as pb2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule
import google.protobuf.text_format
from .common_utils import get_function_name
import os


def as_blob(array):
    blob = pb2.BlobProto()
    blob.shape.dim.extend(array.shape)
    blob.data.extend(array.astype(float).flat)
    return blob


class PytorchCaffeParser:
    def __init__(self, model, inputs, fuse=False, concrete_args=None):
        super(PytorchCaffeParser, self).__init__()
        self.model = model.eval()
        self.inputs = inputs
        if isinstance(self.inputs, torch.Tensor):
            self.inputs = [self.inputs]
        self.fuse = fuse
        self.concrete_args = concrete_args
        self.whether_3dnet = False

        if isinstance(self.model, GraphModule):
            pass
        elif isinstance(self.model, nn.Module):
            if self.fuse:
                self.fuse_all_conv_bn(self.model)
        else:
            raise Exception(
                "model must be a torch.nn.Module \
                            or a torch.fx.GraphModule"
            )

        self.pytorch_graph = PytorchGraph(self.model, self.inputs, self.concrete_args)
        self.state_dict = self.pytorch_graph.graph_module.state_dict()
        self.modules = dict(self.pytorch_graph.graph_module.named_modules())
        self.layers = []

    def convert(self):
        self.text_net, self.binary_weights = self.gen_ir()

    def save(self, dest_path):
        self.dest_path = dest_path
        self.save_to_proto(self.text_net, dest_path + ".prototxt")
        self.save_weights(self.binary_weights, dest_path + ".caffemodel")
        logger.info(f"prototxt saved to {dest_path}.prototxt")
        logger.info(f"caffemodel saved to {dest_path}.caffemodel")

    def save_to_proto(self, net, filename):
        with open(filename, "wb") as f:
            f.write(google.protobuf.text_format.MessageToString(net).encode())

    def save_weights(self, weights, filename):
        with open(filename, "wb") as f:
            f.write(weights.SerializeToString())

    def fuse_all_conv_bn(self, model):
        stack = []
        for name, module in model.named_children():
            if list(module.named_children()):
                self.fuse_all_conv_bn(module)

            if isinstance(module, nn.BatchNorm2d):
                if not stack:
                    continue
                if isinstance(stack[-1][1], nn.Conv2d):
                    setattr(
                        model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module)
                    )
                    setattr(model, name, nn.Identity())

            elif isinstance(module, nn.BatchNorm1d):
                if not stack:
                    continue
                if isinstance(stack[-1][1], nn.Linear):
                    setattr(
                        model, stack[-1][0], fuse_linear_bn_eval(stack[-1][1], module)
                    )
                    setattr(model, name, nn.Identity())
            else:
                stack.append((name, module))

    def list_try_get(self, list, idx, default=None):
        try:
            return list[idx]
        except IndexError:
            return default

    def find_name(self, node):
        if node.op == "placeholder":
            return node.name
        elif node.op == "call_module":
            module = self.modules[node.target]
            if isinstance(module, nn.Identity):
                node_ = node.args[0]
                return self.find_name(node_)
            else:
                return node.name
        elif node.op == "call_function":
            function_name = get_function_name(node.target)
            if function_name == "getitem":
                if isinstance(node.args[1], int):
                    node_name = node.args[0].name + "_" + str(node.args[1])
                    return node_name
                else:
                    return node.name
            else:
                return node.name
        elif node.op == "call_method":
            if str(node.target) == "contiguous":
                node_ = node.args[0]
                return self.find_name(node_)
            else:
                return node.name

    def add_bottom_top(self, layer, source_node):
        for node in source_node.args:
            if isinstance(node, Node):
                bottom_name = self.find_name(node)
                if bottom_name is None:
                    continue
                layer.bottom.append(bottom_name)
            elif isinstance(node, list) or isinstance(node, tuple):
                for node_ in node:
                    if isinstance(node_, Node):
                        bottom_name = self.find_name(node_)
                        if bottom_name is None:
                            continue
                        layer.bottom.append(bottom_name)
            else:
                continue
        layer.top.append(source_node.name)
        layer.name = source_node.name

    def remove_layer(self, bottom=None):
        for layer in self.layers:
            if bottom is not None and bottom in layer.top:
                pre_bottom = layer.bottom[0]
                self.layers.remove(layer)
                for pre_layer in self.layers:
                    if pre_bottom is not None and pre_bottom in pre_layer.top:
                        self.layers.remove(pre_layer)
                        return pre_layer

    def gen_ir(self):
        text_net = pb2.NetParameter()
        binary_weights = pb2.NetParameter()
        for node in self.pytorch_graph.nodes:
            if node.op == "placeholder":
                self.rename_Data(node, text_net)
                # layer_data = self.rename_Data(node)
                # self.layers.append(layer_data)
            elif node.op == "call_module":
                module = self.modules[node.target]
                if isinstance(module, nn.Conv2d):
                    layer_data = self.rename_Conv(node, module)
                    self.layers.append(layer_data)
                if isinstance(module, nn.Conv3d):
                    layer_data = self.rename_Conv3d(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    layer_data = self.rename_BatchNormalization(node, module)
                    self.layers.append(layer_data[0])
                    self.layers.append(layer_data[1])
                elif isinstance(module, nn.BatchNorm3d):
                    layer_data = self.rename_BatchNormalization3d(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, (nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    layer_data = self.rename_InstanceNorm(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.ReLU):
                    layer_data = self.rename_ReLU(node)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.MaxPool2d):
                    layer_data = self.rename_MaxPool2d(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.MaxPool3d):
                    layer_data = self.rename_MaxPool3d(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.AdaptiveAvgPool2d):
                    if isinstance(module.output_size, int):
                        output_size = [1]
                        output_size_len = 1
                    else:
                        output_size = [int(v) for v in module.output_size]
                        output_size_len = len(module.output_size)
                    if output_size == [1] * output_size_len:
                        layer_data = self.rename_AdaptiveAvgPool2d(node)
                    else:
                        layer_data = self.rename_AveragePool(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.AdaptiveMaxPool2d):
                    if isinstance(module.output_size, int):
                        output_size = [1]
                        output_size_len = 1
                    else:
                        output_size = [int(v) for v in module.output_size]
                        output_size_len = len(module.output_size)
                    if output_size == [1] * output_size_len:
                        layer_data = self.rename_AdaptiveMaxPool2d(node)
                    else:
                        layer_data = self.rename_MaxPool(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Linear):
                    layer_data = self.rename_Linear(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Dropout):
                    layer_data = self.rename_Dropout(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.ReLU6):
                    layer_data = self.rename_ReLU6(node)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Hardswish):
                    layer_data = self.rename_Hardswish(node)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Hardsigmoid):
                    layer_data = self.rename_Hardsigmoid(node)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Identity):
                    pass
                elif isinstance(module, nn.AvgPool2d):
                    layer_data = self.rename_AveragePool(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.SiLU):
                    layer_data = self.rename_SiLU(node, module)
                    self.layers.append(layer_data[0])
                    self.layers.append(layer_data[1])
                elif isinstance(module, nn.Upsample):
                    layer_data = self.rename_Upsample(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.LeakyReLU):
                    layer_data = self.rename_LeakyRelu(node, module)
                    self.layers.append(layer_data)
                elif str(module) == "L2Norm()":
                    layer_data = self.rename_L2Norm(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.ConvTranspose2d):
                    layer_data = self.rename_ConvTranspose(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.ConvTranspose3d):
                    layer_data = self.rename_ConvTranspose3d(node, module)
                    self.layers.append(layer_data)
                elif isinstance(module, nn.Sigmoid):
                    layer_data = self.rename_sigmoid(node)
                    self.layers.append(layer_data)
                else:
                    raise NotImplementedError("module %s is not implemented" % (module))
            elif node.op == "call_function":
                function_name = get_function_name(node.target)
                if function_name == "relu":
                    layer_data = self.rename_relu(node)
                    self.layers.append(layer_data)
                elif function_name == "add":
                    layer_data = self.rename_add(node)
                    self.layers.append(layer_data)
                elif function_name == "flatten":
                    layer_data = self.rename_flatten(node)
                    self.layers.append(layer_data)
                elif function_name == "cat":
                    layer_data = self.rename_cat(node)
                    self.layers.append(layer_data)
                elif function_name == "adaptive_avg_pool2d":
                    if isinstance(node.args[1], int):
                        output_size = [1]
                        output_size_len = 1
                    else:
                        output_size = [int(v) for v in node.args[1]]
                        output_size_len = len(node.args[1])
                    if output_size == [1] * output_size_len:
                        layer_data = self.rename_adaptive_avg_pool2d(node)
                    else:
                        layer_data = self.rename_avg_pool2d(node)
                    self.layers.append(layer_data)
                elif function_name == "hardsigmoid":
                    layer_data = self.rename_hardsigmoid(node)
                    self.layers.append(layer_data)
                elif function_name == "mul":
                    layer_data = self.rename_mul(node)
                    if isinstance(layer_data, tuple):
                        for layer in layer_data:
                            self.layers.append(layer)
                    elif layer_data is None:
                        pass
                    else:
                        self.layers.append(layer_data)
                elif function_name == "getitem":
                    if isinstance(node.args[1], tuple):
                        if all(
                            isinstance(function, slice) for function in node.args[1]
                        ):
                            for idx, function in enumerate(node.args[1]):
                                if (
                                    function.start is None
                                    and function.stop is None
                                    and function.step is None
                                ):
                                    continue
                                else:
                                    start_ = (
                                        function.start
                                        if function.start is not None
                                        else 0
                                    )
                                    end_ = (
                                        function.stop
                                        if function.stop is not None
                                        else 1
                                    )  # maybe a bug
                                    axes_ = idx
                                    step_ = (
                                        function.step
                                        if function.step is not None
                                        else 1
                                    )

                                    params_slice = [
                                        np.array([start_]),
                                        np.array([end_]),
                                        np.array([axes_]),
                                        np.array([step_]),
                                    ]
                                    layer_data = self.rename_Slice(node, params_slice)
                                    self.layers.append(layer_data)
                    else:
                        layer_data = self.rename_getitem(node) #fist add, last remove
                        self.layers.append(layer_data)
                elif function_name == "floordiv":
                    pass
                elif function_name == "transpose":
                    layer_data = self.rename_transpose(node)
                    self.layers.append(layer_data)
                elif function_name == "prelu":
                    layer_data = self.rename_prelu(node)
                    self.layers.append(layer_data)
                elif function_name == "tanh":
                    layer_data = self.rename_tanh(node)
                    self.layers.append(layer_data)
                elif function_name == "hardtanh":
                    layer_data = self.rename_hardtanh(node)
                    self.layers.append(layer_data)
                elif function_name == "leaky_relu":
                    layer_data = self.rename_leaky_relu(node)
                    self.layers.append(layer_data)
                elif function_name == "sigmoid":
                    layer_data = self.rename_sigmoid(node)
                    self.layers.append(layer_data)
                elif function_name == "softmax":
                    layer_data = self.rename_softmax(node)
                    self.layers.append(layer_data)
                elif function_name == "hardswish":
                    layer_data = self.rename_hardswish(node)
                    self.layers.append(layer_data)
                elif function_name == "conv2d":
                    layer_data = self.rename_conv2d(node)
                    self.layers.append(layer_data)
                elif function_name == "linear":
                    layer_data = self.rename_linear(node)
                    self.layers.append(layer_data)
                elif function_name == "avg_pool2d":
                    layer_data = self.rename_avg_pool2d(node)
                    self.layers.append(layer_data)
                elif function_name == "max_pool2d_with_indices":
                    layer_data = self.rename_max_pool2d_with_indices(node)
                    self.layers.append(layer_data)
                elif function_name == "chunk":
                    layer_data = self.rename_chunk(node)
                    self.layers.append(layer_data)
                elif function_name == "split":
                    layer_data = self.rename_split(node)
                    self.layers.append(layer_data)
                elif function_name == "sub":
                    layer_data = self.rename_sub(node)
                    self.layers.append(layer_data)
                elif function_name == "abs":
                    layer_data = self.rename_abs(node)
                    self.layers.append(layer_data)
                elif function_name == "getattr": # first add, last remove
                    layer_data = self.rename_getattr(node)
                    self.layers.append(layer_data)
                elif function_name == "interpolate":
                    layer_data = self.rename_interpolate(node)
                    self.layers.append(layer_data)
                else:
                    raise NotImplementedError(
                        "function %s is not implemented" % (function_name)
                    )
            elif node.op == "call_method":
                if str(node.target) == "size":
                    pass
                elif str(node.target) == "view":
                    layer_data = self.rename_view(node)
                    self.layers.append(layer_data)
                elif str(node.target) == "contiguous":
                    pass
                elif str(node.target) == "chunk":
                    layer_data = self.rename_chunk(node)
                    self.layers.append(layer_data)
                elif str(node.target) == "mean":
                    layer_data = self.rename_adaptive_avg_pool2d(node)
                    self.layers.append(layer_data)
                elif str(node.target) == "permute":
                    layer_data = self.rename_permute(node)
                    self.layers.append(layer_data)
                elif str(node.target) == "flatten":
                    layer_data = self.rename_view(node)
                    self.layers.append(layer_data)
                elif str(node.target) == "sigmoid":
                    layer_data = self.rename_sigmoid(node)
                    self.layers.append(layer_data)
                elif str(node.target) == "squeeze":
                    layer_data = self.rename_view(node)
                    self.layers.append(layer_data)
                elif str(node.target) == "transpose":
                    layer_data = self.rename_transpose(node)
                    self.layers.append(layer_data)
                elif str(node.target) == "split":
                    layer_data = self.rename_split(node)
                    self.layers.append(layer_data)
                else:
                    raise NotImplementedError(
                        "method %s is not implemented" % (str(node.target))
                    )
            elif node.op == "output":
                pass
            elif node.op == "get_attr":
                pass
            else:
                raise NotImplementedError("op type %s is not implemented" % (node.op))

        binary_weights.CopyFrom(text_net)

        for layer in self.layers:
            layer.name = layer.name.replace(".", "")
            binary_weights.layer.extend([layer])
            layer_proto = pb2.LayerParameter()
            layer_proto.CopyFrom(layer)
            del layer_proto.blobs[:]
            text_net.layer.extend([layer_proto])

        # print(text_net)
        return text_net, binary_weights

    def pyotrch_inference(self, generate_onnx=False):
        self.pytorch_output = self.model(*self.inputs)

        if isinstance(self.pytorch_output, torch.Tensor):
            self.pytorch_output = [self.pytorch_output]

    def caffe_inference(self):
        prototxt = self.dest_path + ".prototxt"
        caffemodel = self.dest_path + ".caffemodel"

        ppl_dir = './ppl_dir'
        if not os.path.exists(ppl_dir):
            os.makedirs(ppl_dir)
        os.chdir(ppl_dir)

        input_data = np.array([], np.float32)
        if isinstance(self.inputs, (tuple, list)):
            for idx, _ in enumerate(self.inputs):
                img = self.inputs[idx].numpy()
                input_data = np.append(input_data, img.reshape(-1))
        else:
            img = self.inputs[0].numpy()
            input_data = np.append(input_data, img.reshape(-1))
        input_data.tofile("input.bin")

        rm_cmd = "rm ./*dat"
        down_cmd = "wget http://ppl.sensetime.com/shared/packages/GitLab-CI/PPLv3/wjf/develop.2d70805c3fbff4163661e07bd4ef1410056913f2.755276/linux-x86_64+cuda11_1/gcc-stdc++/5.4/Release/pkg.zip"
        unzip_cmd = "unzip ./pkg.zip"
        binary_path = "."
        if os.path.exists("/mnt/hpc/share/wjf/PPLBinaries/PPLV3-CUDA-CARE/pkg.zip"):
            down_cmd = ""
            unzip_cmd = ""
            binary_path = "/mnt/hpc/share/wjf/PPLBinaries/PPLV3-CUDA-CARE"
        pplc_cmd = "{}/bin/pplc --use-cuda -proto-txt ../{} -model ../{}".format(binary_path, prototxt, caffemodel)
        pplr_cmd = "{}/bin/pplr -model ./pplmodel -input ./input.bin --save-output".format(binary_path)
        cmds = [rm_cmd, down_cmd, unzip_cmd, pplc_cmd, pplr_cmd]
        for cmd in cmds:
            os.system(cmd)
        f_list = os.listdir(".")
        import re
        self.caffe_output = []
        for fl in f_list:
            if re.search("ppl3_output-", fl):
                self.caffe_output.append(np.fromfile(fl, np.float32))
        # self.caffe_output = [self.caffe_output[-1], self.caffe_output[0]]

    def check_result(self):
        self.pyotrch_inference()
        self.caffe_inference()
        assert len(self.pytorch_output) == len(
            self.caffe_output
        ), "pytorch_output: %d vs caffe_output %d" % (
            len(self.pytorch_output),
            len(self.caffe_output),
        )

        for idx in range(len(self.caffe_output)):
            np.testing.assert_allclose(
                self.caffe_output[idx],
                self.pytorch_output[idx].detach().numpy().reshape(-1),
                rtol=1e-7,
                atol=1e-3,
            )
        logger.info("accuracy test passed")

    def export_onnx(self, name, opset_version=13):
        self.dummy_input = self.gen_pytorch_input_tensor(self.input_shape)
        torch.onnx.export(
            self.model,
            tuple(self.dummy_input),
            name,
            opset_version=opset_version,
            enable_onnx_checker=False,
        )

    def rename_Data(self, source_node, text_net):
        text_net.input.append(source_node.name)
        input_shape = pb2.BlobShape()
        input_shape.dim.extend(source_node.meta["tensor_meta"]["shape"])
        text_net.input_shape.append(input_shape)

    def add_inputs(self):
        inps = [inp for inp in self.graph.input if inp not in self.graph.initializer]
        self.caffe_net.input.extend(inps)
        for inp in inps:
            inp_shape = pb2.BlobShape()
            input_shape.dim.extend(source_node.meta["tensor_meta"]["shape"])
            inp_shape.dim.extend(self.graph.get_tensor_shape(inp))

    def add_input_layers(self):
        for ipt in self.graph.network_inputs:
            ipt_layer = self.caffe_net.layer.add()
            ipt_layer.name = ipt
            ipt_layer.type = "Input"
            ipt_layer.top[:] = [ipt]
            inp_shape = ipt_layer.input_param.shape.add()
            inp_shape.dim[:] = self.graph.get_tensor_shape(ipt)

    def rename_Conv(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Convolution"

        dilation = module.dilation
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        groups = module.groups

        if isinstance(dilation, tuple):
            if dilation[0] == dilation[1]:
                layer.convolution_param.hole = dilation[0]
            else:
                layer.convolution_param.hole_h = dilation[0]
                layer.convolution_param.hole_w = dilation[1]
        else:
            layer.convolution_param.hole = dilation

        if isinstance(padding, tuple):
            if padding[0] == padding[1]:
                layer.convolution_param.pad = padding[0]
            else:
                layer.convolution_param.pad_h = padding[0]
                layer.convolution_param.pad_w = padding[1]
        else:
            layer.convolution_param.pad = padding

        if isinstance(stride, tuple):
            if stride[0] == stride[1]:
                layer.convolution_param.stride = stride[0]
            else:
                layer.convolution_param.stride_h = stride[0]
                layer.convolution_param.stride_w = stride[1]
        else:
            layer.convolution_param.stride = stride

        if isinstance(kernel_size, tuple):
            if kernel_size[0] == kernel_size[1]:
                layer.convolution_param.kernel_size = kernel_size[0]
            else:
                layer.convolution_param.kernel_h = kernel_size[0]
                layer.convolution_param.kernel_w = kernel_size[1]
        else:
            layer.convolution_param.kernel_size = kernel_size

        layer.convolution_param.group = groups

        weight = module.weight.detach().numpy()

        layer.convolution_param.num_output = module.out_channels

        if module.bias is not None:
            bias = module.bias.detach().numpy()
            layer.convolution_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.convolution_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Conv3d(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Convolution3d"

        dilation = module.dilation
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        groups = module.groups

        if isinstance(dilation, tuple):
            if dilation[0] == dilation[1] and dilation[1] == dilation[2]:
                layer.convolution3d_param.hole = dilation[0]
            else:
                layer.convolution3d_param.hole_d = dilation[0]
                layer.convolution3d_param.hole_h = dilation[1]
                layer.convolution3d_param.hole_w = dilation[2]
        else:
            layer.convolution3d_param.hole = dilation

        if isinstance(padding, tuple):
            if padding[0] == padding[1] and padding[1] == padding[2]:
                layer.convolution3d_param.pad = padding[0]
            else:
                layer.convolution3d_param.pad_d = padding[0]
                layer.convolution3d_param.pad_h = padding[1]
                layer.convolution3d_param.pad_w = padding[2]
        else:
            layer.convolution3d_param.pad = padding

        if isinstance(stride, tuple):
            if stride[0] == stride[1] and stride[1] == stride[2]:
                layer.convolution3d_param.stride = stride[0]
            else:
                layer.convolution3d_param.stride_d = stride[0]
                layer.convolution3d_param.stride_h = stride[1]
                layer.convolution3d_param.stride_w = stride[2]
        else:
            layer.convolution3d_param.stride = stride

        if isinstance(kernel_size, tuple):
            if kernel_size[0] == kernel_size[1] and kernel_size[1] == kernel_size[2]:
                layer.convolution3d_param.kernel_size = kernel_size[0]
            else:
                layer.convolution3d_param.kernel_d = kernel_size[0]
                layer.convolution3d_param.kernel_h = kernel_size[1]
                layer.convolution3d_param.kernel_w = kernel_size[2]
        else:
            layer.convolution3d_param.kernel_size = kernel_size

        layer.convolution3d_param.group = groups

        weight = module.weight.detach().numpy()

        layer.convolution3d_param.num_output = module.out_channels

        if module.bias is not None:
            bias = module.bias.detach().numpy()
            layer.convolution3d_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.convolution3d_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        self.add_bottom_top(layer, source_node)

        self.whether_3dnet = True
        return layer

    def rename_AdaptiveAvgPool2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_AdaptiveMaxPool2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.MAX
        layer.pooling_param.global_pooling = True

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_BatchNormalization(self, source_node, module):
        layer_bn = pb2.LayerParameter()
        layer_bn.type = "BatchNorm"

        layer_bn.batch_norm_param.use_global_stats = 1
        layer_bn.batch_norm_param.eps = module.eps

        mean = module.running_mean.detach().numpy()
        variance = module.running_var.detach().numpy()

        layer_bn.blobs.extend(
            [as_blob(mean), as_blob(variance), as_blob(np.array([1.0]))]
        )

        layer_bn.bottom.append(source_node.args[0].name)

        layer_bn.top.append(source_node.name + "_bn")
        layer_bn.name = source_node.name + "_bn"

        layer_scale = pb2.LayerParameter()
        layer_scale.type = "Scale"

        weight = module.weight.detach().numpy()

        if module.bias is not None:
            bias = module.bias.detach().numpy()
            layer_scale.scale_param.bias_term = True
            layer_scale.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer_scale.scale_param.bias_term = False
            layer_scale.blobs.extend([as_blob(weight)])

        layer_scale.bottom.append(source_node.name + "_bn")
        layer_scale.top.append(source_node.name)
        layer_scale.name = source_node.name

        return layer_bn, layer_scale

    def rename_BatchNormalization3d(self, source_node, module):
        layer_bn = pb2.LayerParameter()

        weight = module.weight.detach().numpy()
        bias = module.bias.detach().numpy()
        mean = module.running_mean.detach().numpy()
        variance = module.running_var.detach().numpy()

        # if module.track_running_stats:
        if False:
            layer_bn.bn3d_param.moving_average = False
            layer_bn.bn3d_param.var_eps = module.eps
            layer_bn.bn3d_param.decay = module.momentum
            layer_bn.type = "BN3d"
            layer_bn.blobs.extend(
                [as_blob(weight), as_blob(bias), as_blob(mean), as_blob(variance)]
            )
        else:
            layer_bn.batchnorm3d_param.use_global_stats = True
            layer_bn.type = "BatchNorm3d"
            layer_bn.batchnorm3d_param.eps = module.eps
            layer_bn.blobs.extend(
                [as_blob(weight), as_blob(bias), as_blob(mean), as_blob(variance)]
            )

        self.add_bottom_top(layer_bn, source_node)

        return layer_bn

    def rename_InstanceNorm(self, source_node, module):
        layer_in = pb2.LayerParameter()
        layer_in.type = "InstanceNorm"

        layer_in.instance_norm_param.affine = module.affine
        layer_in.instance_norm_param.eps = module.eps
        layer_in.instance_norm_param.num_features = module.num_features

        if module.affine:
            weight = module.weight.detach().numpy()
            if module.bias is not None:
                bias = module.bias.detach().numpy()
                layer_in.blobs.extend([as_blob(weight), as_blob(bias)])
            else:
                layer_in.blobs.extend([as_blob(weight)])

        self.add_bottom_top(layer_in, source_node)

        return layer_in

    def rename_ReLU(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"
        if self.whether_3dnet:
            layer.type = "ReLU3d"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_MaxPool2d(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.MAX

        if isinstance(module.padding, tuple):
            if module.padding[0] == module.padding[1]:
                layer.pooling_param.pad = module.padding[0]
            else:
                layer.pooling_param.pad_h = module.padding[0]
                layer.pooling_param.pad_w = module.padding[1]
        else:
            layer.pooling_param.pad = module.padding

        if isinstance(module.stride, tuple):
            if module.stride[0] == module.stride[1]:
                layer.pooling_param.stride = module.stride[0]
            else:
                layer.pooling_param.stride_h = module.stride[0]
                layer.pooling_param.stride_w = module.stride[1]
        else:
            layer.pooling_param.stride = module.stride

        if isinstance(module.kernel_size, tuple):
            if module.kernel_size[0] == module.kernel_size[1]:
                layer.pooling_param.kernel_size = module.kernel_size[0]
            else:
                layer.pooling_param.kernel_h = module.kernel_size[0]
                layer.pooling_param.kernel_w = module.kernel_size[1]
        else:
            layer.pooling_param.kernel_size = module.kernel_size

        layer.pooling_param.ceil_mode = module.ceil_mode
        layer.pooling_param.global_pooling = False

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_MaxPool3d(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Pooling3d"

        layer.pooling3d_param.pool = pb2.PoolingParameter.MAX

        if isinstance(module.padding, tuple):
            if module.padding[0] == module.padding[1] and module.padding[1] == module.padding[2]:
                layer.pooling3d_param.pad = module.padding[0]
            else:
                layer.pooling3d_param.pad_d = module.padding[0]
                layer.pooling3d_param.pad_h = module.padding[1]
                layer.pooling3d_param.pad_w = module.padding[2]
        else:
            layer.pooling3d_param.pad = module.padding

        if isinstance(module.stride, tuple):
            if module.stride[0] == module.stride[1] and module.stride[1] == module.stride[2]:
                layer.pooling3d_param.stride = module.stride[0]
            else:
                layer.pooling3d_param.stride_d = module.stride[0]
                layer.pooling3d_param.stride_h = module.stride[1]
                layer.pooling3d_param.stride_w = module.stride[2]
        else:
            layer.pooling3d_param.stride = module.stride

        if isinstance(module.kernel_size, tuple):
            if module.kernel_size[0] == module.kernel_size[1] and module.kernel_size[1] == module.kernel_size[2]:
                layer.pooling3d_param.kernel_size = module.kernel_size[0]
            else:
                layer.pooling3d_param.kernel_d = module.kernel_size[0]
                layer.pooling3d_param.kernel_h = module.kernel_size[1]
                layer.pooling3d_param.kernel_w = module.kernel_size[2]
        else:
            layer.pooling3d_param.kernel_size = module.kernel_size

        layer.pooling3d_param.ceil_mode = module.ceil_mode
        layer.pooling3d_param.global_pooling = False

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Add(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Eltwise"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_AveragePool(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        if isinstance(module, nn.AdaptiveAvgPool2d):
            dim = source_node.meta["tensor_meta"]["shape"][2:]
            if isinstance(module.output_size, int):
                output_size = [module.output_size] * len(dim)
            else:
                output_size = module.output_size
            k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
            if len(k) == 1:
                layer.pooling_param.stride = k[0]
            else:
                if k[0] == k[1]:
                    layer.pooling_param.stride = k[0]
                else:
                    layer.pooling_param.stride_h = k[0]
                    layer.pooling_param.stride_w = k[1]

            if len(k) == 1:
                layer.pooling_param.kernel_size.extend(k[0])
            else:
                if k[0] == k[1]:
                    layer.pooling_param.kernel_size = k[0]
                else:
                    layer.pooling_param.kernel_h = k[0]
                    layer.pooling_param.kernel_w = k[1]
            self.add_bottom_top(layer, source_node)

            return layer
        else:
            padding = module.padding
            stride = module.stride
            kernel_size = module.kernel_size
            ceil_mode = module.ceil_mode
            if isinstance(padding, tuple):
                if padding[0] == padding[1]:
                    layer.pooling_param.pad.extend([padding[0]])
                else:
                    layer.pooling_param.pad_h = padding[0]
                    layer.pooling_param.pad_w = padding[1]
            else:
                layer.pooling_param.pad = padding

            if isinstance(stride, tuple):
                if stride[0] == stride[1]:
                    layer.pooling_param.stride.extend([stride[0]])
                else:
                    layer.pooling_param.stride_h = stride[0]
                    layer.pooling_param.stride_w = stride[1]
            else:
                layer.pooling_param.stride = stride

            if isinstance(kernel_size, tuple):
                if kernel_size[0] == kernel_size[1]:
                    layer.pooling_param.kernel_size.extend([kernel_size[0]])
                else:
                    layer.pooling_param.kernel_h = kernel_size[0]
                    layer.pooling_param.kernel_w = kernel_size[1]
            else:
                layer.pooling_param.kernel_size = kernel_size

            layer.pooling_param.ceil_mode = ceil_mode

            self.add_bottom_top(layer, source_node)

            return layer

    def rename_Flatten(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Flatten"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Linear(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "InnerProduct"

        weight = module.weight.detach().numpy()

        if module.bias is not None:
            bias = module.bias.detach().numpy()
            layer.inner_product_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.inner_product_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        layer.inner_product_param.num_output = module.out_features

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Dropout(self, source_node, module):
        layer = pb2.LayerParameter()
        if self.whether_3dnet:
            layer.type = "Dropout3d"
            layer.dropout3d_param.dropout_ratio = module.p
        else:
            layer.type = "Dropout"
            layer.dropout_param.dropout_ratio = module.p

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Permute(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Permute"

        for arg in source_node.args:
            if isinstance(arg, int):
                layer.permute_param.order.extend([arg])

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Upsample(self, source_node, module=None):
        layer = pb2.LayerParameter()
        if module.mode == "nearest":
            layer.type = "NNUpsample"
            layer.nn_upsample_param.resize = int(module.scale_factor)
        elif self.whether_3dnet and module.mode == "trilinear":
            layer.type = "Interp3d"
            layer.interp3d_param.zoom_factor = int(module.scale_factor)
            layer.interp3d_param.align_corners = module.align_corners
            layer.interp3d_param.backend = pb2.Interp3dParameter.ModeType.pytorch
        elif module.mode == "bilinear":
            layer.type = "Interp"
            layer.interp_param.zoom_factor = int(module.scale_factor)
            layer.interp_param.align_corners = module.align_corners
            layer.interp_param.backend = pb2.InterpParameter.ModeType.pytorch
        else:
            assert 0, "Upsample convert failed!"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_interpolate(self, source_node, module=None):
        layer = pb2.LayerParameter()

        use_factor = True
        self.add_bottom_top(layer, source_node)
        for idx in range(len(source_node.kwargs["size"])):
            item = source_node.kwargs["size"][idx].name
            bottom_layer = self.remove_layer(item)
        if len(bottom_layer.bottom) > 0:
            use_factor = False
            layer.bottom.append(bottom_layer.bottom[0])

        scale_factor = 1
        if isinstance(source_node.kwargs["scale_factor"], tuple):
            scale_factor = source_node.kwargs["scale_factor"][0]
        elif source_node.kwargs["scale_factor"] is not None:
            scale_factor = int(source_node.kwargs["scale_factor"])

        mode = source_node.kwargs["mode"]
        align_corners = source_node.kwargs["align_corners"]
        if mode == "nearest":
            layer.type = "NNUpsample"
            layer.nn_upsample_param.resize = int(scale_factor)
        elif self.whether_3dnet and mode == "trilinear":
            layer.type = "Interp3d"
            if use_factor:
                layer.interp3d_param.zoom_factor = int(scale_factor)
            layer.interp3d_param.align_corners = align_corners
            layer.interp3d_param.backend = pb2.Interp3dParameter.ModeType.pytorch
        elif mode == "bilinear":
            layer.type = "Interp"
            if use_factor:
                layer.interp_param.zoom_factor = int(scale_factor)
            layer.interp_param.align_corners = module.align_corners
            layer.interp_param.backend = pb2.InterpParameter.ModeType.pytorch
        else:
            assert 0, "Interpolate convert failed!"

        

        return layer

    def rename_Cat(self, source_node):
        layer = pb2.LayerParameter()

        axis = 1
        if "dim" in source_node.kwargs:
            axis = source_node.kwargs["dim"]
        else:
            axis = source_node.args[1]

        if self.whether_3dnet:
            layer.type = "Concat3d"
            layer.concat3d_param.axis = axis
        else:
            layer.type = "Concat"
            layer.concat_param.axis = axis

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_ReLU6(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU6"

        layer.relu6_param.threshold = 6

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Hardswish(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "HardSwish"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Hardsigmoid(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "HardSigmoid"
        layer.hardsigmoid_param.alpha = 1.0 / 6
        layer.hardsigmoid_param.beta = 0.5

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Mul(self, source_node):
        def add_flatten_before_mul(node_name, first_input, second_input):
            # first input (N,C,H,W); first input (N,C)
            layer_flatten = pb2.LayerParameter()
            layer_flatten.type = "Flatten"
            layer_flatten.flatten_param.axis = 1
            layer_flatten.bottom.append(second_input.name)
            layer_flatten.top.append(node_name + "_flatten")
            layer_flatten.name = node_name + "_flatten"

            layer_scale = pb2.LayerParameter()
            layer_scale.type = "Scale"
            layer_scale.scale_param.axis = 0
            layer_scale.bottom.append(first_input.name)
            layer_scale.bottom.append(node_name + "_flatten")
            layer_scale.top.append(node_name)
            layer_scale.name = node_name

            return layer_flatten, layer_scale

        shape = list(source_node.args[0].meta["tensor_meta"]["shape"])
        if shape[-2:] == [1, 1]:
            return add_flatten_before_mul(
                source_node.name, source_node.args[1], source_node.args[0]
            )
        elif shape[-2:] == [1, 1]:
            return add_flatten_before_mul(
                source_node.name, source_node.args[0], source_node.args[1]
            )
        else:
            layer = pb2.LayerParameter()
            layer.type = "Scale"
            layer.scale_param.axis = 0
            self.add_bottom_top(layer, source_node)

            return layer

    def rename_view(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Reshape"

        for shape in source_node.meta["tensor_meta"]["shape"]:
            layer.reshape_param.shape.dim.extend([shape])

        bottom_name = self.find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.name = source_node.name

        return layer

    def rename_Split(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        if "dim" in source_node.kwargs:
            layer.slice_param.axis = source_node.kwargs["dim"]
        else:
            layer.slice_param.axis = source_node.args[1]

        sum_ = 0
        for idx in range(len(source_node.meta["tensor_meta"]) - 1):
            tensor_meta = source_node.meta["tensor_meta"][idx]
            sum_ = sum_ + tensor_meta["shape"][layer.slice_param.axis]
            layer.slice_param.slice_point.extend([sum_])

        bottom_name = self.find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        for idx in range(len(source_node.meta["tensor_meta"])):
            layer.top.append(source_node.name + "_" + str(idx))
        layer.name = source_node.name

        return layer

    def rename_L2Norm(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Normalize"

        layer.norm_param.across_spatial = False
        layer.norm_param.scale_filler.type = "constant"
        layer.norm_param.scale_filler.value = module.gamma
        layer.norm_param.channel_shared = False

        weight = module.weight.detach().numpy().squeeze()
        layer.blobs.extend([as_blob(weight)])

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_LeakyRelu(self, source_node, module):
        layer = pb2.LayerParameter()
        if self.whether_3dnet:
            layer.type = "LeakyReLU3d"
            layer.leaky_relu3d_param.negative_slope = module.negative_slope
        else:
            layer.type = "ReLU"
            layer.relu_param.negative_slope = module.negative_slope

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_SiLU(self, source_node, module):
        layer_sigmoid = pb2.LayerParameter()
        layer_sigmoid.type = "Sigmoid"
        layer_sigmoid.bottom.append(self.find_name(source_node.args[0]))

        layer_sigmoid.top.append(source_node.name + "_sigmoid")
        layer_sigmoid.name = source_node.name + "_sigmoid"

        layer_scale = pb2.LayerParameter()
        layer_scale.type = "Scale"
        layer_scale.scale_param.axis = 0

        layer_scale.bottom.append(self.find_name(source_node.args[0]))
        layer_scale.bottom.append(source_node.name + "_sigmoid")
        layer_scale.top.append(source_node.name)
        layer_scale.name = source_node.name

        return layer_sigmoid, layer_scale

    def rename_hardtanh(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU6"

        layer.relu6_param.threshold = source_node.kwargs["max_val"]

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_flatten(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Flatten"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_relu(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"
        if self.whether_3dnet:
            layer.type = "ReLU3d"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_add(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Eltwise"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_sub(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Eltwise"
        layer.eltwise_param.operation = 1  # sum
        layer.eltwise_param.coeff.extend([1, -1])

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_abs(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "AbsVal"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_getattr(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "GetAttr"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_getitem(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "GetItem"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_cat(self, source_node):
        layer = pb2.LayerParameter()
        axis = 1
        if "dim" in source_node.kwargs:
            axis = source_node.kwargs["dim"]
        else:
            axis = source_node.args[1]

        if self.whether_3dnet:
            layer.type = "Concat3d"
            layer.concat3d_param.axis = axis
        else:
            layer.type = "Concat"
            layer.concat_param.axis = axis

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_adaptive_avg_pool2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_adaptive_max_pool2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.MAX
        layer.pooling_param.global_pooling = True

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_hardsigmoid(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "HardSigmoid"
        layer.hardsigmoid_param.alpha = 1.0 / 6
        layer.hardsigmoid_param.beta = 0.5

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_mul(self, source_node):
        def add_flatten_before_mul(node_name, first_input, second_input):
            # first input (N,C,H,W); second input (N,C)
            layer_flatten = pb2.LayerParameter()
            layer_flatten.type = "Flatten"
            layer_flatten.flatten_param.axis = 1
            layer_flatten.bottom.append(self.find_name(second_input))
            layer_flatten.top.append(node_name + "_flatten")
            layer_flatten.name = node_name + "_flatten"

            layer_scale = pb2.LayerParameter()
            layer_scale.type = "Scale"
            layer_scale.scale_param.axis = 0
            layer_scale.bottom.append(self.find_name(first_input))
            layer_scale.bottom.append(node_name + "_flatten")
            layer_scale.top.append(node_name)
            layer_scale.name = node_name

            return layer_flatten, layer_scale

        def add_tile_before_mul(node_name, first_input, second_input):
            # first input (N,C,H,W); second input (N,1,H,W)
            layer_flatten = pb2.LayerParameter()
            layer_flatten.type = "Tile"
            layer_flatten.tile_param.axis = 1
            layer_flatten.tile_param.tiles = list(
                first_input.meta["tensor_meta"]["shape"]
            )[1]
            layer_flatten.bottom.append(self.find_name(second_input))
            layer_flatten.top.append(node_name + "_tile")
            layer_flatten.name = node_name + "_tile"

            layer_eltwise = pb2.LayerParameter()
            layer_eltwise.type = "Eltwise"
            layer_eltwise.eltwise_param.operation = 0  # prod
            layer_eltwise.bottom.append(self.find_name(first_input))
            layer_eltwise.bottom.append(node_name + "_tile")
            layer_eltwise.top.append(node_name)
            layer_eltwise.name = node_name

            return layer_flatten, layer_eltwise

        if "tensor_meta" not in list(source_node.args[0].meta.keys()):
            return

        shape_0 = list(source_node.args[0].meta["tensor_meta"]["shape"])
        shape_1 = list(source_node.args[1].meta["tensor_meta"]["shape"])

        if shape_0[-2:] == shape_1[-2:]:
            if shape_0[1] == shape_1[1]:
                layer = pb2.LayerParameter()
                layer.type = "Scale"
                layer.scale_param.axis = 0
                self.add_bottom_top(layer, source_node)

                return layer
            elif shape_0[1] == 1:
                return add_tile_before_mul(
                    source_node.name, source_node.args[1], source_node.args[0]
                )
            elif shape_1[1] == 1:
                return add_tile_before_mul(
                    source_node.name, source_node.args[0], source_node.args[1]
                )
            else:
                raise NotImplementedError("unsupported shape for mul")
        elif shape_0[-2:] == [1, 1]:
            return add_flatten_before_mul(
                source_node.name, source_node.args[1], source_node.args[0]
            )
        elif shape_1[-2:] == [1, 1]:
            return add_flatten_before_mul(
                source_node.name, source_node.args[0], source_node.args[1]
            )
        else:
            layer = pb2.LayerParameter()
            layer.type = "Scale"
            layer.scale_param.axis = 0
            self.add_bottom_top(layer, source_node)

            return layer

    def rename_permute(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Permute"

        for arg in source_node.args:
            if isinstance(arg, int):
                layer.permute_param.order.extend([arg])

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_leaky_relu(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "ReLU"

        layer.relu_param.negative_slope = source_node.kwargs["negative_slope"]

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_sigmoid(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Sigmoid"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_softmax(self, source_node):
        layer = pb2.LayerParameter()

        if "dim" in source_node.kwargs.keys():
            dim = source_node.kwargs["dim"]
        else:
            dim = None
        if dim is None:
            stacklevel = 3
            shape = source_node.args[0].meta["tensor_meta"]["shape"]
            dim = F._get_softmax_dim("softmax", len(shape), stacklevel)

        if self.whether_3dnet:
            layer.type = "Softmax3d"
            layer.softmax3d_param.axis = dim
        else:
            layer.type = "Softmax"
            layer.softmax_param.axis = dim

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_hardswish(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "HardSwish"

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_conv2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Convolution"

        weight = self.model.weight.detach().numpy()
        bias = self.model.bias
        stride = source_node.args[3]
        padding = source_node.args[4]
        dilation = source_node.args[5]
        groups = source_node.args[6]

        layer.convolution_param.dilation.extend([dilation[0]])

        if isinstance(padding, tuple):
            if padding[0] == padding[1]:
                layer.convolution_param.pad.extend([padding[0]])
            else:
                layer.convolution_param.pad_h = padding[0]
                layer.convolution_param.pad_w = padding[1]
        else:
            layer.convolution_param.pad.extend([padding])

        if isinstance(stride, tuple):
            if stride[0] == stride[1]:
                layer.convolution_param.stride.extend([stride[0]])
            else:
                layer.convolution_param.stride_h = stride[0]
                layer.convolution_param.stride_w = stride[1]
        else:
            layer.convolution_param.stride.extend([stride])

        if weight.shape[2] == weight.shape[3]:
            layer.convolution_param.kernel_size.extend([weight.shape[2]])
        else:
            layer.convolution_param.kernel_h = weight.shape[2]
            layer.convolution_param.kernel_w = weight.shape[3]

        layer.convolution_param.group = groups

        layer.convolution_param.num_output = weight.shape[0]

        if bias is not None:
            bias = bias.detach().numpy()
            layer.convolution_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.convolution_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        bottom_name = self.find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.name = source_node.name

        return layer

    def rename_linear(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "InnerProduct"

        weight = self.model.weight.detach().numpy()
        bias = self.model.bias

        if bias is not None:
            bias = bias.detach().numpy()
            layer.inner_product_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.inner_product_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        layer.inner_product_param.num_output = weight.shape[0]

        bottom_name = self.find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.name = source_node.name

        return layer

    def rename_avg_pool2d(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.AVE
        function_name = get_function_name(source_node.target)
        if function_name == "adaptive_avg_pool2d":
            output_size = source_node.args[1]
            dim = source_node.args[0].meta["tensor_meta"]["shape"][2:]
            if isinstance(output_size, int):
                output_size = [output_size] * len(dim)
            else:
                output_size = output_size
            k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
            if len(k) == 1:
                layer.pooling_param.stride = k[0]
            else:
                if k[0] == k[1]:
                    layer.pooling_param.stride = k[0]
                else:
                    layer.pooling_param.stride_h = k[0]
                    layer.pooling_param.stride_w = k[1]

            if len(k) == 1:
                layer.pooling_param.kernel_size.extend(k[0])
            else:
                if k[0] == k[1]:
                    layer.pooling_param.kernel_size = k[0]
                else:
                    layer.pooling_param.kernel_h = k[0]
                    layer.pooling_param.kernel_w = k[1]

            self.add_bottom_top(layer, source_node)

            return layer
        else:
            kernel_size = source_node.args[1]
            stride = self.list_try_get(source_node.args, 2, kernel_size)
            padding = self.list_try_get(source_node.args, 3, 0)
            ceil_mode = self.list_try_get(source_node.args, 4, False)

            if isinstance(padding, tuple):
                if padding[0] == padding[1]:
                    layer.pooling_param.pad.extend([padding[0]])
                else:
                    layer.pooling_param.pad_h = padding[0]
                    layer.pooling_param.pad_w = padding[1]
            else:
                layer.pooling_param.pad = padding

            if isinstance(stride, tuple):
                if stride[0] == stride[1]:
                    layer.pooling_param.stride = stride[0]
                else:
                    layer.pooling_param.stride_h = stride[0]
                    layer.pooling_param.stride_w = stride[1]
            else:
                layer.pooling_param.stride = stride

            if isinstance(kernel_size, tuple):
                if kernel_size[0] == kernel_size[1]:
                    layer.pooling_param.kernel_size = kernel_size[0]
                else:
                    layer.pooling_param.kernel_h = kernel_size[0]
                    layer.pooling_param.kernel_w = kernel_size[1]
            else:
                layer.pooling_param.kernel_size = kernel_size

            layer.pooling_param.ceil_mode = ceil_mode

            bottom_name = self.find_name(source_node.args[0])
            layer.bottom.append(bottom_name)
            layer.top.append(source_node.name)
            layer.name = source_node.name

            return layer

    def rename_max_pool2d_with_indices(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Pooling"

        layer.pooling_param.pool = pb2.PoolingParameter.MAX

        kernel_size = source_node.args[1]
        stride = source_node.kwargs["stride"]
        padding = source_node.kwargs["padding"]
        dilation = source_node.kwargs["dilation"]
        ceil_mode = source_node.kwargs["ceil_mode"]
        return_indices = source_node.kwargs["return_indices"]

        if isinstance(padding, tuple):
            if padding[0] == padding[1]:
                layer.pooling_param.pad = padding[0]
            else:
                layer.pooling_param.pad_h = padding[0]
                layer.pooling_param.pad_w = padding[1]
        else:
            layer.pooling_param.pad = padding

        if isinstance(stride, tuple):
            if stride[0] == stride[1]:
                layer.pooling_param.stride = stride[0]
            else:
                layer.pooling_param.stride_h = stride[0]
                layer.pooling_param.stride_w = stride[1]
        else:
            layer.pooling_param.stride = stride

        if isinstance(kernel_size, tuple):
            if kernel_size[0] == kernel_size[1]:
                layer.pooling_param.kernel_size = kernel_size[0]
            else:
                layer.pooling_param.kernel_h = kernel_size[0]
                layer.pooling_param.kernel_w = kernel_size[1]
        else:
            layer.pooling_param.kernel_size = kernel_size

        layer.pooling_param.ceil_mode = ceil_mode

        if return_indices:
            bottom_name = self.find_name(source_node.args[0])
            layer.bottom.append(bottom_name)
            layer.top.append(source_node.name + "_" + str(0))
            layer.top.append(source_node.name + "_" + str(1))
            layer.name = source_node.name
        else:
            self.add_bottom_top(layer, source_node)

        return layer

    def rename_chunk(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Slice"
        if "dim" in source_node.kwargs:
            layer.slice_param.axis = source_node.kwargs["dim"]
        else:
            layer.slice_param.axis = source_node.args[2]

        sum_ = 0
        for idx in range(len(source_node.meta["tensor_meta"]) - 1):
            tensor_meta = source_node.meta["tensor_meta"][idx]
            sum_ = sum_ + tensor_meta["shape"][layer.slice_param.axis]
            layer.slice_param.slice_point.extend([sum_])

        bottom_name = self.find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        for idx in range(len(source_node.meta["tensor_meta"])):
            layer.top.append(source_node.name + "_" + str(idx))
        layer.name = source_node.name

        return layer

    def rename_split(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        if "dim" in source_node.kwargs:
            layer.slice_param.axis = source_node.kwargs["dim"]
        else:
            layer.slice_param.axis = source_node.args[2]

        sum_ = 0
        for idx in range(len(source_node.meta["tensor_meta"]) - 1):
            tensor_meta = source_node.meta["tensor_meta"][idx]
            sum_ = sum_ + tensor_meta["shape"][layer.slice_param.axis]
            layer.slice_param.slice_point.extend([sum_])

        bottom_name = self.find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        for idx in range(len(source_node.meta["tensor_meta"])):
            layer.top.append(source_node.name + "_" + str(idx))
        layer.name = source_node.name

        return layer

    def rename_transpose(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "Permute"

        input_dim = len(source_node.args[0].meta["tensor_meta"]["shape"])
        axes = list(range(input_dim))
        axes[source_node.args[1]], axes[source_node.args[2]] = (
            axes[source_node.args[2]],
            axes[source_node.args[1]],
        )
        for axe in axes:
            layer.permute_param.order.extend([axe])

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_prelu(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "PReLU"

        weight = self.model.weight.detach().numpy()

        layer.prelu_param.channel_shared = True
        layer.blobs.extend([as_blob(weight[0])])

        bottom_name = self.find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.name = source_node.name

        return layer

    def rename_ConvTranspose(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Deconvolution"

        dilation = module.dilation
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        groups = module.groups

        if isinstance(dilation, tuple):
            if dilation[0] == dilation[1]:
                layer.convolution_param.hole = dilation[0]
            else:
                layer.convolution_param.hole_h = dilation[0]
                layer.convolution_param.hole_w = dilation[1]
        else:
            layer.convolution_param.hole = dilation

        if isinstance(padding, tuple):
            if padding[0] == padding[1]:
                layer.convolution_param.pad = padding[0]
            else:
                layer.convolution_param.pad_h = padding[0]
                layer.convolution_param.pad_w = padding[1]
        else:
            layer.convolution_param.pad = padding

        if isinstance(stride, tuple):
            if stride[0] == stride[1]:
                layer.convolution_param.stride = stride[0]
            else:
                layer.convolution_param.stride_h = stride[0]
                layer.convolution_param.stride_w = stride[1]
        else:
            layer.convolution_param.stride = stride

        if isinstance(kernel_size, tuple):
            if kernel_size[0] == kernel_size[1]:
                layer.convolution_param.kernel_size = kernel_size[0]
            else:
                layer.convolution_param.kernel_h = kernel_size[0]
                layer.convolution_param.kernel_w = kernel_size[1]
        else:
            layer.convolution_param.kernel_size = kernel_size

        layer.convolution_param.group = groups

        weight = module.weight.detach().numpy()

        layer.convolution_param.num_output = module.out_channels

        if module.bias is not None:
            bias = module.bias.detach().numpy()
            layer.convolution_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.convolution_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        self.add_bottom_top(layer, source_node)

        return layer
    
    def rename_ConvTranspose3d(self, source_node, module):
        layer = pb2.LayerParameter()
        layer.type = "Deconvolution3d"

        dilation = module.dilation
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        out_padding = module.output_padding
        groups = module.groups

        if isinstance(dilation, tuple):
            if dilation[0] == dilation[1] and dilation[1] == dilation[2]:
                layer.deconvolution3d_param.hole = dilation[0]
            else:
                layer.deconvolution3d_param.hole_d = dilation[0]
                layer.deconvolution3d_param.hole_h = dilation[1]
                layer.deconvolution3d_param.hole_w = dilation[2]
        else:
            layer.deconvolution3d_param.hole = dilation

        if isinstance(padding, tuple):
            if padding[0] == padding[1] and padding[1] == padding[2]:
                layer.deconvolution3d_param.pad = padding[0]
            else:
                layer.deconvolution3d_param.pad_d = padding[0]
                layer.deconvolution3d_param.pad_h = padding[1]
                layer.deconvolution3d_param.pad_w = padding[2]
        else:
            layer.deconvolution3d_param.pad = padding

        if isinstance(out_padding, tuple):
            if out_padding[0] == out_padding[1] and out_padding[1] == out_padding[2]:
                layer.deconvolution3d_param.out_pad = out_padding[0]
            else:
                layer.deconvolution3d_param.out_pad_d = out_padding[0]
                layer.deconvolution3d_param.out_pad_h = out_padding[1]
                layer.deconvolution3d_param.out_pad_w = out_padding[2]
        else:
            layer.deconvolution3d_param.out_pad = out_padding

        if isinstance(stride, tuple):
            if stride[0] == stride[1] and stride[1] == stride[2]:
                layer.deconvolution3d_param.stride = stride[0]
            else:
                layer.deconvolution3d_param.stride_d = stride[0]
                layer.deconvolution3d_param.stride_h = stride[1]
                layer.deconvolution3d_param.stride_w = stride[2]
        else:
            layer.deconvolution3d_param.stride = stride

        if isinstance(kernel_size, tuple):
            if kernel_size[0] == kernel_size[1] and kernel_size[1] == kernel_size[2]:
                layer.deconvolution3d_param.kernel_size = kernel_size[0]
            else:
                layer.deconvolution3d_param.kernel_d = kernel_size[0]
                layer.deconvolution3d_param.kernel_h = kernel_size[1]
                layer.deconvolution3d_param.kernel_w = kernel_size[2]
        else:
            layer.deconvolution3d_param.kernel_size = kernel_size

        layer.deconvolution3d_param.group = groups

        weight = module.weight.detach().numpy()

        layer.deconvolution3d_param.num_output = module.out_channels

        if module.bias is not None:
            bias = module.bias.detach().numpy()
            layer.deconvolution3d_param.bias_term = True
            layer.blobs.extend([as_blob(weight), as_blob(bias)])
        else:
            layer.deconvolution3d_param.bias_term = False
            layer.blobs.extend([as_blob(weight)])

        self.add_bottom_top(layer, source_node)

        return layer

    def rename_Slice(self, source_node, slice_params):
        layer = pb2.LayerParameter()
        layer.type = "Slice"

        layer.slice_param.axis = slice_params[2]

        layer.slice_param.slice_point.extend([slice_params[1]])

        bottom_name = self.find_name(source_node.args[0])
        layer.bottom.append(bottom_name)
        layer.top.append(source_node.name)
        layer.top.append(source_node.name + "_index1")
        layer.name = source_node.name

        return layer

    def rename_tanh(self, source_node):
        layer = pb2.LayerParameter()
        layer.type = "TanH"

        self.add_bottom_top(layer, source_node)

        return layer

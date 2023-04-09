using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Runtime.InteropServices;
using System.Linq;
using System.IO;
using System;

public class FastOnnxLoader : MonoBehaviour
{
    InferenceSession session = null;
    OrtIoBinding binding;
    private void FastLoadTest() {
        if (session != null)
        {
            session.Dispose();
        }
        Debug.Log("Begin...");

        SessionOptions sessionOptions = new SessionOptions
        {
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            EnableMemoryPattern = false,
        };
        sessionOptions.AppendExecutionProvider_DML(0);

        session = new InferenceSession("model_only.onnx", sessionOptions);
        sessionOptions.Dispose();
        Debug.Log("session=" + session);


        //---now we load all the weights from the files and bind them to the inputs
        binding = session.CreateIoBinding();
        Float16[] float16s = null;
        bool[] bools = null;
        float[] floats = null;
        SByte[] sbytes = null;
        int[] ints = null;

        foreach (var key in session.InputMetadata.Keys)
        {
            int[] dims = session.InputMetadata[key].Dimensions;
            string fname= key;
            fname = fname.Replace(":", "_");
            string filename = "weights\\" + fname;
            string eType = session.InputMetadata[key].ElementType.Name;

            var eName = session.InputMetadata[key].ElementType.Name;
            if (dims[0] > 0 && File.Exists(filename))
            {
                byte[] bytes = File.ReadAllBytes(filename);

                switch (eName)
                {
                    case "Boolean":
                        bools = new bool[bytes.Length];
                        Buffer.BlockCopy(bytes, 0, bools, 0, bytes.Length);
                        using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<bool>(bools, dims)))
                        {
                            binding.BindInput(key, value); binding.SynchronizeBoundInputs();
                        }
                        break;
                    case "Byte":
                        using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<byte>(bytes, dims)))
                        {
                            binding.BindInput(key, value); binding.SynchronizeBoundInputs();
                        }
                        break;
                    case "SByte":
                        sbytes = BytesTo<SByte>(bytes);
                        using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<SByte>(sbytes, dims)))
                        {
                            binding.BindInput(key, value); binding.SynchronizeBoundInputs();
                        }
                        break;
                    case "Float16":
                        float16s = BytesTo<Float16>(bytes);
                        using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<Float16>(float16s, dims)))
                        //using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<Float16>(new Memory<Float16>(float16s,0,dimSize), dims)))
                        {
                            binding.BindInput(key, value); binding.SynchronizeBoundInputs();
                        }
                        break;
                    case "Single":
                        floats = BytesTo<float>(bytes);
                        using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<float>(floats, dims)))
                        {
                            binding.BindInput(key, value); binding.SynchronizeBoundInputs();
                        }
                        break;
                    case "Int":
                        ints = BytesTo<int>(bytes);
                        using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<int>(ints, dims)))
                        {
                            binding.BindInput(key, value); binding.SynchronizeBoundInputs();
                        }
                        break;
                    default:
                        Debug.Log("Type not found:" + session.InputMetadata[key].ElementType.Name);
                        return (session, null);
                }
                float16s = null;
                bools = null;
                ints = null;
                floats = null;
            }
            else
            {
                Debug.Log("File not found:" + key + "\n" + string.Join(",", dims) + "=" + TensorExt.DimSize(dims) * 2 + ":" + session.InputMetadata[key].ElementType.Name);
            }
        }

        //---Testing inference----
        for (int i = 1; i < 3; i++)
        {
            using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<long>(new int[] { 1, i })))
            {
                binding.BindInput("input_ids", value); binding.SynchronizeBoundInputs();
            }
            using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<Float16>(new int[] { 1, i, 50257 })))
            {
                binding.BindOutput("output", value); binding.SynchronizeBoundOutputs();
            }
            var output = session.RunWithBindingAndNames(new RunOptions { }, binding);
            var tensor = output.First().AsTensor<Float16>();
            Debug.Log("output=" + string.Join(",", tensor.Dimensions.ToArray()));
            var floats = TensorExt.ToFloat(tensor).ToArray<float>(); //custom function to turn Float16 Tensor to float tensor

            Debug.Log(floats[0] + "," + floats[1] + ",...");
        }

        session.Dispose();
        Debug.Log("End...");
    }

    private void OnApplicationQuit()
    {
        if (session != null)
        {
            session.Dispose();
        }
    }
    
    //Fast Byte conversion
      public static T[] BytesTo<T>(byte[] source)
     {
         T[] floats = new T[source.Length /  Marshal.SizeOf(typeof(T))];

         GCHandle handle = GCHandle.Alloc(floats, GCHandleType.Pinned);
         try
         {
             System.IntPtr pointer = handle.AddrOfPinnedObject();
             Marshal.Copy(source, 0, pointer, source.Length);
             return floats;
         }
         finally
         {
             if (handle.IsAllocated)
                 handle.Free();
         }
     }
}


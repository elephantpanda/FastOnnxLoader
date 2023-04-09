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

        foreach (var key in session.InputMetadata.Keys)
        {
            int[] dims = session.InputMetadata[key].Dimensions;
            string filename = "weights\\" + key;
            string eType = session.InputMetadata[key].ElementType.Name;

            if (File.Exists(filename))
            {
                long length = new FileInfo(filename).Length;
                byte[] bytes = File.ReadAllBytes(filename);
                
                if ( eType == "Boolean")
                {
                    bools = new bool[bytes.Length];
                    Buffer.BlockCopy(bytes, 0, bools, 0, bytes.Length);
                }
                else if (eType == "Float16")
                {
                    float16s = BytesToFloat16(bytes);
                }
                else
                {
                    Debug.Log("Error:Unknown type:" + session.InputMetadata[key].ElementType.Name);
                    return;
                }
            }
            else
            {
                Debug.Log("Error: Weight file not found!" + key);
                //could be an input
            }
            if (dims[0] > 0){
                if (eType  == "Boolean")
                {
                    using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<bool>(bools, dims)))
                    {
                        binding.BindInput(key, value); binding.SynchronizeBoundInputs();
                    }
                }
                else if (eType == "Float16")
                {
                    using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<Float16>(float16s, dims)))
                    {
                        binding.BindInput(key, value); binding.SynchronizeBoundInputs();
                    }
                }
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
            var floats = TensorExt.ToFloat(tensor).ToArray<float>();

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
    
    //Fast Byte array to Float16 array
     public static Float16[] BytesToFloat16(byte[] source)
    {
        Float16[] float16s = new Float16[source.Length/2];

        GCHandle handle = GCHandle.Alloc(float16s, GCHandleType.Pinned);
        try
        {
            System.IntPtr pointer = handle.AddrOfPinnedObject();
            Marshal.Copy(source, 0, pointer, source.Length);
            return float16s;
        }
        finally
        {
            if (handle.IsAllocated)
                handle.Free();
        }
    }
}


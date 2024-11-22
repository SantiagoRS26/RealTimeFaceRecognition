using DAL.Interfaces;
using Emgu.CV.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Servicios
{
    public class FaceModelLoader : IFaceModelLoader
    {
        public Net LoadModel(string modelConfiguration, string modelWeights)
        {
            return DnnInvoke.ReadNetFromCaffe(modelConfiguration, modelWeights);
        }
    }
}

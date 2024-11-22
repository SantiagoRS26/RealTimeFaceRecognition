using Emgu.CV.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Interfaces
{
    public interface IFaceModelLoader
    {
        Net LoadModel(string modelConfiguration, string modelWeights);
    }
}

using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Interfaces
{
    public interface IFaceDetectionService : IDisposable
    {
        Rectangle[] DetectFaces(Mat frame);
    }
}

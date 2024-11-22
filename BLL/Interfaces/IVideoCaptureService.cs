using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Interfaces
{
    public interface IVideoCaptureService : IDisposable
    {
        event EventHandler<Mat> FrameCaptured;

        void Start();

        void Stop();

        double GetFramesPerSecond();
        int GetFrameWidth();
        int GetFrameHeight();

    }
}

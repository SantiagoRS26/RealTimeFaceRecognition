using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models.Domain
{
    public class Detection
    {
        public int DetectionId { get; set; }
        public int IntervalId { get; set; }
        public DateTime Timestamp { get; set; }
        public float PositionX { get; set; }
        public float PositionY { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
        public float Confidence { get; set; }

        public Interval Interval { get; set; }
    }
}

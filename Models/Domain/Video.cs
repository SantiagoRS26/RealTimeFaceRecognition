using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models.Domain
{
    public class Video
    {
        public int VideoId { get; set; }
        public string FilePath { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public int DurationInSeconds { get; set; }

        public ICollection<Interval> Intervals { get; set; }
        public ICollection<Log> Logs { get; set; }
    }
}

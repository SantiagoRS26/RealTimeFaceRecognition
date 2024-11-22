using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models.Domain
{
    public class Interval
    {
        public int IntervalId { get; set; }
        public int? VideoId { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public int MaxPersons { get; set; }
        public float AveragePersons { get; set; }
        public string Notes { get; set; }

        public Video Video { get; set; }
        public ICollection<Detection> Detections { get; set; }
        public ICollection<Log> Logs { get; set; }
    }
}

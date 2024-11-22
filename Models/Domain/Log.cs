using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models.Domain
{
    public class Log
    {
        public int LogId { get; set; }
        public DateTime LogTimestamp { get; set; }
        public string Level { get; set; }
        public string Event { get; set; }
        public int? VideoId { get; set; }
        public int? IntervalId { get; set; }

        public Video Video { get; set; }
        public Interval Interval { get; set; }
    }
}

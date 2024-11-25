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
        public string FilePath { get; set; } // Ruta del archivo del video
        public DateTime StartTime { get; set; } // Inicio del video
        public DateTime EndTime { get; set; } // Fin del video
        public int MaxPersons { get; set; } // Máximo de personas detectadas
        public float AveragePersons { get; set; } // Promedio de personas detectadas
        public int DurationInSeconds { get; set; } // Duración en segundos del video

        public ICollection<Log> Logs { get; set; } // Relación con Logs

        public string? S3Url { get; set; }
    }
}

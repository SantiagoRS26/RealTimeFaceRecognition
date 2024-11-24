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
        public DateTime Timestamp { get; set; } // Fecha y hora del evento
        public string Event { get; set; } // Descripción del evento (ej. "Persona detectada", "Inicio de grabación")
        public int? VideoId { get; set; } // Relación con Video (opcional)

        public Video Video { get; set; } // Relación con Video
    }
}

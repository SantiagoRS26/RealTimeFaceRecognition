using BLL.Interfaces;
using DAL.Interfaces;
using Models.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Services
{
    public class LogService : ILogService
    {
        private readonly IUnitOfWork _unitOfWork;

        public LogService(IUnitOfWork unitOfWork)
        {
            _unitOfWork = unitOfWork;
        }

        public async Task<IEnumerable<Log>> GetAllLogsAsync()
        {
            return await _unitOfWork.Logs.GetAllAsync();
        }

        public async Task<Log> GetLogByIdAsync(int logId)
        {
            return await _unitOfWork.Logs.GetByIdAsync(logId);
        }

        public async Task<IEnumerable<Log>> GetLogsByVideoIdAsync(int videoId)
        {
            return await _unitOfWork.Logs.GetAsync(log => log.VideoId == videoId);
        }

        public async Task AddLogAsync(Log log)
        {
            if (log == null)
                throw new ArgumentNullException(nameof(log));

            // Validar que los campos requeridos estén presentes
            if (string.IsNullOrWhiteSpace(log.Event))
                throw new ArgumentException("El campo 'Event' es obligatorio.", nameof(log.Event));

            if (log.Timestamp == default)
                throw new ArgumentException("El campo 'Timestamp' debe contener una fecha y hora válidas.", nameof(log.Timestamp));

            try
            {
                await _unitOfWork.Logs.AddAsync(log);
                await _unitOfWork.CommitAsync();

                Console.WriteLine($"Log añadido correctamente: {log.Event}");
            }
            catch (Exception ex)
            {
                // Manejo de excepciones y registro
                Console.WriteLine($"Error al guardar el log: {ex.Message}");
                throw new InvalidOperationException("Ocurrió un error al guardar el log en la base de datos.", ex);
            }
        }


        public async Task UpdateLogAsync(Log log)
        {
            if (log == null)
                throw new ArgumentNullException(nameof(log));

            _unitOfWork.Logs.Update(log);
            await _unitOfWork.CommitAsync();
        }

        public async Task DeleteLogAsync(int logId)
        {
            var log = await _unitOfWork.Logs.GetByIdAsync(logId);
            if (log == null)
                throw new KeyNotFoundException($"Log with ID {logId} not found.");

            _unitOfWork.Logs.Remove(log);
            await _unitOfWork.CommitAsync();
        }
    }
}
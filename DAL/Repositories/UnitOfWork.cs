using DAL.Context;
using DAL.Interfaces;
using Models.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DAL.Repositories
{
    public class UnitOfWork : IUnitOfWork
    {
        private readonly ApplicationDbContext _context;
        private IGenericRepository<Video> _videos;
        private IGenericRepository<Log> _logs;

        public UnitOfWork(ApplicationDbContext context)
        {
            _context = context;
        }

        public IGenericRepository<Video> Videos => _videos ??= new GenericRepository<Video>(_context);
        public IGenericRepository<Log> Logs => _logs ??= new GenericRepository<Log>(_context);

        public async Task<int> CommitAsync()
        {
            return await _context.SaveChangesAsync();
        }

        public void Dispose()
        {
            _context.Dispose();
        }
    }
}

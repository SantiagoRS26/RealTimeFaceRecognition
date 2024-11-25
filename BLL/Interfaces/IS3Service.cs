using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BLL.Interfaces
{
    public interface IS3Service
    {
        Task<string> UploadFileAsync(string filePath, string keyName);
    }
}

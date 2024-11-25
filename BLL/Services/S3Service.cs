using Amazon;
using Amazon.S3;
using Amazon.S3.Transfer;
using BLL.Interfaces;
using System;
using System.Threading.Tasks;

namespace BLL.Services
{
    public class S3Service : IS3Service
    {
        private readonly IAmazonS3 _s3Client;
        private readonly string _bucketName;

        public S3Service(string awsAccessKey, string awsSecretKey, string awsRegion, string bucketName)
        {
            _bucketName = bucketName;

            var awsRegionEndpoint = RegionEndpoint.GetBySystemName(awsRegion);

            _s3Client = new AmazonS3Client(awsAccessKey, awsSecretKey, awsRegionEndpoint);
        }

        public async Task<string> UploadFileAsync(string filePath, string keyName)
        {
            try
            {
                var fileTransferUtility = new TransferUtility(_s3Client);

                // Subir el archivo sin usar ACLs
                var uploadRequest = new TransferUtilityUploadRequest
                {
                    FilePath = filePath,
                    BucketName = _bucketName,
                    Key = keyName
                };

                await fileTransferUtility.UploadAsync(uploadRequest);

                // Construir la URL pública del archivo
                string url = $"https://{_bucketName}.s3.{_s3Client.Config.RegionEndpoint.SystemName}.amazonaws.com/{keyName}";
                return url;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Error al subir el archivo a S3: {ex.Message}", ex);
            }
        }
    }
}

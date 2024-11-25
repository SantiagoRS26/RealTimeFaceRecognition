using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace DAL.Migrations
{
    /// <inheritdoc />
    public partial class OnCascade : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Logs_Videos_VideoId",
                table: "Logs");

            migrationBuilder.AddForeignKey(
                name: "FK_Logs_Videos_VideoId",
                table: "Logs",
                column: "VideoId",
                principalTable: "Videos",
                principalColumn: "VideoId",
                onDelete: ReferentialAction.Cascade);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Logs_Videos_VideoId",
                table: "Logs");

            migrationBuilder.AddForeignKey(
                name: "FK_Logs_Videos_VideoId",
                table: "Logs",
                column: "VideoId",
                principalTable: "Videos",
                principalColumn: "VideoId");
        }
    }
}

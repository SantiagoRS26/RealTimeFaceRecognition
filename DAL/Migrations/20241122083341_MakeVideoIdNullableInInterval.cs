using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace DAL.Migrations
{
    /// <inheritdoc />
    public partial class MakeVideoIdNullableInInterval : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Intervals_Videos_VideoId",
                table: "Intervals");

            migrationBuilder.AlterColumn<int>(
                name: "VideoId",
                table: "Intervals",
                type: "integer",
                nullable: true,
                oldClrType: typeof(int),
                oldType: "integer");

            migrationBuilder.AddForeignKey(
                name: "FK_Intervals_Videos_VideoId",
                table: "Intervals",
                column: "VideoId",
                principalTable: "Videos",
                principalColumn: "VideoId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Intervals_Videos_VideoId",
                table: "Intervals");

            migrationBuilder.AlterColumn<int>(
                name: "VideoId",
                table: "Intervals",
                type: "integer",
                nullable: false,
                defaultValue: 0,
                oldClrType: typeof(int),
                oldType: "integer",
                oldNullable: true);

            migrationBuilder.AddForeignKey(
                name: "FK_Intervals_Videos_VideoId",
                table: "Intervals",
                column: "VideoId",
                principalTable: "Videos",
                principalColumn: "VideoId",
                onDelete: ReferentialAction.Cascade);
        }
    }
}

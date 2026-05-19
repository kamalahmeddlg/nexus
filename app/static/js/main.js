document.addEventListener("DOMContentLoaded", function () {

    const imageInput = document.getElementById("imageInput");
    const fileName = document.getElementById("fileName");
    const uploadForm = document.querySelector(".upload-form");
    const loader = document.getElementById("uploadLoader");
    const previewImage = document.getElementById("previewImage");

    if (imageInput) {
        imageInput.addEventListener("change", function () {

            if (this.files.length > 0) {
                const file = this.files[0];

                fileName.innerHTML = file.name;

                const reader = new FileReader();

                reader.onload = function (e) {
                    previewImage.src = e.target.result;
                    previewImage.style.display = "block";
                }

                reader.readAsDataURL(file);
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener("submit", function () {
            loader.style.display = "flex";
        });
    }

});
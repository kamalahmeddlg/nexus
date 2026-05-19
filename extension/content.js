async function checkImage(img) {
    try {
        if (!img.src) return;

        const response = await fetch("http://127.0.0.1:7860/api/check", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image_url: img.src
            })
        });

        const data = await response.json();

        if (data.label === "NSFW") {
            img.style.filter = "blur(35px)";
            img.style.transition = "0.3s";
        }

    } catch (error) {
        console.log(error);
    }
}

function scanImages() {
    const images = document.querySelectorAll("img");

    images.forEach((img) => {
        if (!img.dataset.scanned) {
            img.dataset.scanned = "true";
            checkImage(img);
        }
    });
}

scanImages();

setInterval(scanImages, 3000);
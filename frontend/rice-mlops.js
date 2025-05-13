
function readFileB () {
    // (A) GET SELECTED FILE
    let selected = document.getElementById("demoPickB").files[0];

    // (B) READ SELECTED FILE
    let reader = new FileReader();
    reader.addEventListener("load", () => {
        var data = new FormData();
        data.append("image", reader.result);

        let imgTag = document.getElementById("demoShowB");
        imgTag.src = reader.result;

        var xhr = new XMLHttpRequest();
        xhr.withCredentials = false;

        xhr.open("POST", "http://127.0.0.1:5000/classify");

        xhr.onload = function() {
            let resultTag = document.getElementById("result");
            console.log(this.responseText)
            result = JSON.parse(this.response)
            resultTag.innerHTML = "<p>Label: "+result[0].label+"</p><p>Confidence: "+result[0].prob+"</p><p>Model: "+result[0].model_id+"</p>";
        };

        xhr.send(data);

    });
    reader.readAsDataURL(selected);
}

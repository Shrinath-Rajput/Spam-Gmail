function checkEmail() {
    const emailText = document.getElementById("emailText").value;
    const box = document.getElementById("resultBox");

    if (emailText.trim() === "") {
        box.innerHTML = `
            <div class="alert alert-warning mt-3">
                Please enter email content
            </div>`;
        return;
    }

    box.innerHTML = `
        <div class="alert alert-info mt-3">
            Checking email...
        </div>`;

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ email: emailText })
    })
    .then(res => res.json())
    .then(data => {
        if (data.result === "SPAM") {
            box.innerHTML = `
                <div class="alert alert-danger mt-3">
                    <strong>Result:</strong> 🚨 SPAM EMAIL
                </div>`;
        } else {
            box.innerHTML = `
                <div class="alert alert-success mt-3">
                    <strong>Result:</strong> ✅ NOT SPAM EMAIL
                </div>`;
        }
    })
    .catch(() => {
        box.innerHTML = `
            <div class="alert alert-danger mt-3">
                Server error occurred
            </div>`;
    });
}

async function ask() {
    const question = document.getElementById("question").value;
    const answerElem = document.getElementById("answer");

    if (!question) {
        answerElem.textContent = "Please type a question!";
        return;
    }

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();

        if (data.answer) {
            answerElem.textContent = data.answer;
        } else if (data.error) {
            answerElem.textContent = "Error: " + data.error;
        }
    } catch (err) {
        answerElem.textContent = "Fetch error: " + err;
    }
}


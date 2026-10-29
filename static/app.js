document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const resultBox = document.getElementById("result-box");
    const passengerCountEl = document.getElementById("pred-passengers");
    const badgeEl = document.getElementById("status-badge");

    // Animation function for counting up
    const animateValue = (obj, start, end, duration) => {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // easeOutQuart
            const easeOut = 1 - Math.pow(1 - progress, 4);
            obj.innerHTML = Math.floor(easeOut * (end - start) + start).toLocaleString();
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    };

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Show result box with a smooth entrance
        resultBox.style.display = "block";
        resultBox.style.animation = "none";
        resultBox.offsetHeight; /* trigger reflow */
        resultBox.style.animation = "fadeInUp 0.8s ease-out";

        // Gather Data
        const formData = {
            year: parseInt(document.getElementById("year").value),
            month: document.getElementById("month").value,
            terminal: document.getElementById("terminal").value,
            region: document.getElementById("region").value,
            activity: document.getElementById("activity").value
        };

        const btn = document.querySelector(".submit-btn");
        const originalText = btn.innerText;
        btn.innerText = "Analyzing...";
        btn.disabled = true;

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Prediction failed");
            }

            const data = await response.json();
            
            // Animate parameters
            animateValue(passengerCountEl, 0, data.prediction, 1500);

            // Update Badge Design
            badgeEl.innerText = data.congestion_level;
            badgeEl.className = "status-badge"; // reset classes
            
            if (data.congestion_level === "Low") {
                badgeEl.classList.add("status-low");
            } else if (data.congestion_level === "Moderate") {
                badgeEl.classList.add("status-moderate");
            } else {
                badgeEl.classList.add("status-high");
            }

        } catch (error) {
            console.error(error);
            passengerCountEl.innerText = "Error";
            badgeEl.innerText = "N/A";
            badgeEl.className = "status-badge";
            alert("Could not complete prediction: " + error.message);
        } finally {
            btn.innerText = originalText;
            btn.disabled = false;
        }
    });
});

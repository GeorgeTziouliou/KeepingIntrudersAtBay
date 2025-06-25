/**
 * Frontend JavaScript for the Smart Home Security interface.
 *
 * Handles UI updates for door status, RFID card detection, face match notifications,
 * system settings toggles, and interactions for manual door control and RFID operations.
 */

const doorToggle = document.getElementById('doorToggle');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const rfidResult = document.getElementById('rfidResult');
const writeModal = document.getElementById('writeModal');
const writeDataInput = document.getElementById('writeData');
const livenessToggle = document.getElementById('livenessToggle');
const allowOtherToggle = document.getElementById('allowOtherToggle');

let currentRFIDName = null;
let matchNotificationTimeout = null;
let useLiveness = true;
let allowOther = false;
let doorStatusInterval = null;

/**
 * Fetches initial door status and system settings when the page loads.
 * Starts polling intervals for door status, RFID name, and match status.
 */
window.addEventListener('DOMContentLoaded', () => {
    fetchDoorStatus();
    fetchSettings();
    startRFIDNameCheck();
    startMatchStatusCheck();
    startDoorStatusPolling();
});

/**
 * Fetches the current system settings (liveness, allow_other) from the server
 * and updates the UI toggles accordingly.
 */
function fetchSettings() {
    fetch('/system_settings')
        .then(response => response.json())
        .then(data => {
            useLiveness = data.use_liveness;
            allowOther = data.allow_other;

            livenessToggle.checked = useLiveness;
            allowOtherToggle.checked = allowOther;
        })
        .catch(error => {
            console.error('Error fetching system settings:', error);
        });
}

/**
 * Handles the change event for the liveness detection toggle.
 * Updates the local state and sends the change to the server.
 */
function toggleLiveness() {
    useLiveness = livenessToggle.checked;
    updateSystemSettings();
}

/**
 * Handles the change event for the 'allow other' category toggle.
 * Updates the local state and sends the change to the server.
 */
function toggleAllowOther() {
    allowOther = allowOtherToggle.checked;
    updateSystemSettings();
}

/**
 * Sends the current state of the system settings toggles to the server.
 * Re-fetches settings on error to ensure UI consistency.
 */
function updateSystemSettings() {
    fetch('/update_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            use_liveness: useLiveness,
            allow_other: allowOther
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("Settings updated successfully");
        } else {
            console.error("Failed to update settings:", data.error);
            fetchSettings(); // Re-sync UI on failure
        }
    })
    .catch(error => {
        console.error('Error updating settings:', error);
        fetchSettings(); // Re-sync UI on failure
    });
}

/**
 * Starts an interval timer to periodically fetch the name/data of the
 * currently detected RFID card from the server and update the display.
 */
function startRFIDNameCheck() {
    setInterval(() => {
        // Use the endpoint from CameraAndWebsite.py which calls door_controller
        fetch('/current_rfid_name')
            .then(response => response.json())
            .then(data => {
                if (data.name !== currentRFIDName) {
                    currentRFIDName = data.name;
                    updateRFIDDisplay();
                }
            })
            .catch(error => {
                console.error('Error fetching RFID name:', error);
                // Handle case where card might be removed causing null name
                if (currentRFIDName !== null) {
                    currentRFIDName = null;
                    updateRFIDDisplay();
                }
            });
    }, 2000); // Check every 2 seconds
}


/**
 * Starts an interval timer to periodically check the face match status
 * from the server and display a notification if a match is found.
 */
function startMatchStatusCheck() {
    setInterval(() => {
        fetch('/match_status')
            .then(response => response.json())
            .then(data => {
                if (data.match_found) {
                    showMatchNotification(data.person_name, data.confidence);
                }
            })
            .catch(error => {
                console.error('Error checking match status:', error);
            });
    }, 1000); // Check every second
}

/**
 * Displays a temporary notification popup when a face match occurs.
 *
 * @param {string} personName - The name of the matched person.
 * @param {number} confidence - The confidence score of the match.
 */
function showMatchNotification(personName, confidence) {
    let matchNotification = document.getElementById('matchNotification');

    if (!matchNotification) {
        matchNotification = document.createElement('div');
        matchNotification.id = 'matchNotification';
        matchNotification.className = 'match-notification';
        document.body.appendChild(matchNotification);
    }

    matchNotification.innerHTML = `
        <div class="match-content">
            <h3>Match Found!</h3>
            <p>Name: ${personName}</p>
            <p>Confidence: ${confidence.toFixed(2)}%</p>
            <p>Door unlocking...</p>
        </div>
    `;

    matchNotification.style.display = 'flex';

    if (matchNotificationTimeout) {
        clearTimeout(matchNotificationTimeout);
    }

    matchNotificationTimeout = setTimeout(() => {
        matchNotification.style.display = 'none';
    }, 5000);

    // Fetch updated door status slightly after notification appears
    setTimeout(fetchDoorStatus, 1000);
}


/**
 * Updates the RFID information display area based on whether a card
 * is currently detected.
 */
function updateRFIDDisplay() {
    if (currentRFIDName) {
        rfidResult.innerHTML = `
            <strong>Current RFID Card:</strong><br>
            Name/Data: ${currentRFIDName}
            <p class="rfid-info">Waiting for face match...</p>
        `;
        rfidResult.classList.add('active-scan');
    } else {
        rfidResult.innerHTML = `
            <strong>No RFID card detected</strong>
            <p class="rfid-info text-danger">An RFID card is required for all access</p>
        `;
        rfidResult.classList.remove('active-scan');
    }
}

/**
 * Starts an interval timer to periodically fetch the door's lock status
 * from the server and update the UI.
 */
function startDoorStatusPolling() {
    if (doorStatusInterval) {
        clearInterval(doorStatusInterval);
    }
    doorStatusInterval = setInterval(() => {
        fetchDoorStatus();
    }, 3000); // Poll every 3 seconds
}

/**
 * Fetches the current door status from the server and updates the UI.
 */
function fetchDoorStatus() {
    fetch('/door_status')
        .then(response => response.json())
        .then(data => {
            updateDoorUI(data.locked);
            if (doorToggle.checked !== data.locked) {
                 doorToggle.checked = data.locked;
            }
        })
        .catch(error => {
            console.error('Error fetching door status:', error);
        });
}

/**
 * Updates the visual elements (status dot, text) representing the door lock state.
 *
 * @param {boolean} isLocked - True if the door is locked, false otherwise.
 */
function updateDoorUI(isLocked) {
    if (isLocked) {
        statusDot.classList.remove('unlocked');
        statusDot.classList.add('locked');
        statusText.textContent = 'Door Locked';
    } else {
        statusDot.classList.remove('locked');
        statusDot.classList.add('unlocked');
        statusText.textContent = 'Door Unlocked';
    }
}

/**
 * Handles the change event for the manual door lock toggle switch.
 * Sends a request to the server to lock or unlock the door.
 * Temporarily pauses polling during the operation.
 */
function toggleDoor() {
    const isLocked = doorToggle.checked ? 1 : 0;

    if (doorStatusInterval) {
        clearInterval(doorStatusInterval);
        doorStatusInterval = null;
    }

    fetch(`/toggle_door/${isLocked}`, {
        method: 'POST',
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateDoorUI(data.locked);
                // Ensure toggle reflects actual state
                if(doorToggle.checked !== data.locked) {
                    doorToggle.checked = data.locked;
                }
            } else {
                 fetchDoorStatus(); // Re-sync UI on failure
            }
        })
        .catch(error => {
            console.error('Error toggling door:', error);
            // Revert UI change optimistically and re-sync
             doorToggle.checked = !doorToggle.checked;
             fetchDoorStatus();
        })
        .finally(() => {
             // Restart polling after operation completes
             startDoorStatusPolling();
        });
}

/**
 * Sends a request to the server to perform a manual RFID card read
 * and displays the result.
 */
function readRFID() {
    rfidResult.innerHTML = 'Reading RFID card...';

    fetch('/rfid/read')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                rfidResult.innerHTML = `
                    <strong>RFID Card Read Successfully:</strong><br>
                    Card ID: ${data.id}<br>
                    Data: ${data.data}
                `;
                // Optionally update currentRFIDName if needed after manual read
                // currentRFIDName = data.data;
                // updateRFIDDisplay(); // Update main display too
            } else {
                rfidResult.innerHTML = `<strong>Error:</strong> ${data.error}`;
            }
        })
        .catch(error => {
            console.error('Error reading RFID:', error);
            rfidResult.innerHTML = `<strong>Network Error:</strong> ${error.message}`;
        });
}

/**
 * Shows the modal dialog for entering data to write to an RFID card.
 */
function showWriteModal() {
    writeModal.style.display = 'block';
}

/**
 * Hides the RFID write modal dialog.
 */
function closeWriteModal() {
    writeModal.style.display = 'none';
}

/**
 * Sends a request to the server to write the data entered in the modal
 * to an RFID card. Displays the result.
 */
function writeRFID() {
    const data = writeDataInput.value.trim();

    if (!data) {
        rfidResult.innerHTML = '<strong>Error:</strong> Please enter data to write';
        return;
    }

    rfidResult.innerHTML = 'Writing to RFID card...';

    fetch('/rfid/write', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: data })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                rfidResult.innerHTML = `<strong>Success:</strong> ${data.message}`;
                closeWriteModal();
                writeDataInput.value = '';
            } else {
                rfidResult.innerHTML = `<strong>Error:</strong> ${data.error}`;
            }
        })
        .catch(error => {
            console.error('Error writing RFID:', error);
            rfidResult.innerHTML = `<strong>Network Error:</strong> ${error.message}`;
        });
}

/**
 * Closes the RFID write modal if the user clicks outside of its content area.
 */
window.onclick = function(event) {
    if (event.target == writeModal) {
        closeWriteModal();
    }
}
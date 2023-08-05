function updateBottom(buttonInfo) {
    var container = document.querySelector(".bottom-right-bottom-block");
    container.innerHTML = "";
    var rowDiv;
    var buttonsPerRow = 5;
    for (var i = 0; i < buttonInfo.length; i++) {
        if (i % buttonsPerRow === 0) {
            rowDiv = document.createElement("div");
            rowDiv.classList.add("BRB-row");
            container.appendChild(rowDiv);
        }

        var info = buttonInfo[i];
        var newButton = document.createElement("button");
        newButton.classList.add("BRB-custom-button");
        newButton.textContent = info.buttonName;
        newButton.id = info.id;
        newButton.addEventListener("click", (function(button) {
            return function() {
                showButtonSelectionForm(button);
            };
        })(info.button));
        rowDiv.appendChild(newButton);
    }
}

function showButtonSelectionForm(button) {
    fetch('/getButtonOptions/' + button)
        .then(response => response.json())
        .then(optionMap => {
            updateButtonSelectionForm(button,optionMap);
        });
    var buttonSelectionForm = document.getElementById("button-selection");
    buttonSelectionForm.style.display = "flex";
}

function updateButtonSelectionForm(buttonTypeName,optionDataMap) {
    var buttonSelectionOption = document.getElementById("button-selection-option");
    buttonSelectionOption.innerHTML = "";
    buttonSelectionOption.style.columnCount = Math.floor(Object.keys(optionDataMap).length / 18 ) + 1 ;
    Object.keys(optionDataMap).forEach(function(labelValue) {
        var li = document.createElement("li");
        var input = document.createElement("input");
        input.type = "checkbox";
        input.name = "button-selection-option";
        input.value = labelValue;
        input.id = 'ButtonOptions'+toString(labelValue);
        input.checked = true;
        input.classList.add("button-selection-checkbox");

        var label = document.createElement("label");
        label.htmlFor = toString(labelValue);
        label.classList.add("button-selection-checkbox-label");
        label.appendChild(document.createTextNode(optionDataMap[labelValue]));

        li.appendChild(input);
        li.appendChild(label);
        buttonSelectionOption.appendChild(li);
    });

    var buttonSelectionSubmit = document.getElementById("button-selection-submit");
    buttonSelectionSubmit.innerHTML = "";
    var confirmButton = document.createElement("button");
    confirmButton.type = "button";
    confirmButton.textContent = "選擇完畢";
    confirmButton.addEventListener("click", function() {
        var checkboxes = document.querySelectorAll('input[name="button-selection-option"]:checked');
        var selectedOptions = [];
        checkboxes.forEach(function(checkbox) {
            selectedOptions.push(checkbox.value);
        });
        sessionStorage.setItem(buttonTypeName, JSON.stringify(selectedOptions));
        console.log(sessionStorage);
        document.getElementById('button-selection').style.display = 'none';
    });
    buttonSelectionSubmit.appendChild(confirmButton);
}

function updateChart() {
    fetch('/getChatData')
        .then(response => response.json())
        .then(chartInfo => {
            var ctx = document.getElementById('Chart_1').getContext('2d');
            var myChart = new Chart(ctx, chartInfo );
        })
        .catch(error => {
            console.error('Error:', error);
        });
}


// ======================================================= 清除相關內容 =======================================================

window.addEventListener('load', function(){sessionStorage.clear();});

// ======================================================= 清除相關內容 =======================================================

document.addEventListener("DOMContentLoaded", function() {
    updateChart()
    fetch('/getButtonTypes')
        .then(response => response.json())
        .then(buttonTypes => {
            updateBottom(buttonTypes);
        });
});


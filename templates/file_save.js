// Add event listener to form to read file once submitted
const form = document.querySelector("form");

form.addEventListener("submit", e => {
    e.preventDefault();
    // save the file from the input file
    print("here")
    const file = e.target[0].files[0]; // getting the first input of the form then the first file of its files property (array)

    //parse the file with Papa.parse
    Papa.parse(file, {
        header: true,
        complete: function(results) {
            // read the data from results
            const { data } = results;
            //print the data in the console
            console.log({ data })
        }
    });
    form.reset();
})
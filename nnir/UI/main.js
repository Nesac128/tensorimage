function graph(){
    var xs = [1,2,3];
    var ys = [5,6,7];
    var acc = document.getElementById("accuracyChart").getContext('2d');

    var accuracyChart = new Chart(acc, {
                      type: 'line',
                      options: {scales:{yAxes: [{ticks: {beginAtZero: true}}]}},
                      data: {
                          labels: ys,
                          datasets: [
                          {
                              label: 'Original Data',
                              data: xs,
                              borderWidth: 1,
                          }
                        ]
                      },
                  });
}
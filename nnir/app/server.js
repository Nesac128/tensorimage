var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http)

const WebSocket = require('ws')
const sleep = require('sleep')

var epochs = [];
var training_accuracy = [];
var training_cost = [];
var testing_accuracy = [];
var testing_cost = [];

new WebSocket('ws://localhost:2025').onmessage = function(recv){
        data = recv['data'].replace(/'/g, '"')
        data = JSON.parse(data)
        epochs.push(data["epoch"]);
        training_accuracy.push(data["training_accuracy"]);
        training_cost.push(data["training_cost"])
        testing_accuracy.push(data["testing_accuracy"]);
        testing_cost.push(data["testing_cost"])
        console.log(data)
};

app.get('/training', function(req, res){
        res.sendFile('/home/planetgazer8360/PycharmProjects/nnir/nnir/app/templates/training.html')
})

io.sockets.on('connection',function (socket) {
    console.log("A WebSocket connection has been established with the client!")
});

setInterval(function() {
    io.emit('data', {
    "epochs": epochs,
    "training_accuracy": training_accuracy,
    "training_cost": training_cost,
    "testing_accuracy": testing_accuracy,
    "testing_cost": testing_cost
    })}, 2000);

http.listen(2024, function() {
    console.log('Listening on localhost:2024');
});

//app.get('/', function(req, res){
//        res.sendFile('templates/index.html')
//})
//

//
//app.listen(2024, 'localhost')
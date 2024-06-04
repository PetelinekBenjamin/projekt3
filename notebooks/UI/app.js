const express = require('express');
const path = require('path');
const axios = require('axios');


const app = express();
const port = 3000;

// Uporabi statično mapo za serviranje HTML datotek
app.use(express.static('public'));




app.get('/api/metrics1', async (req, res) => {
    try {
        const response = await axios.get('http://api:5000/ovrednotenje1');
        res.send(response.data); // Spremenjeno, da pošilja celoten odgovor
    } catch (error) {
        res.status(500).send('Napaka pri pridobivanju podatkov');
    }
});

app.get('/api/metrics2', async (req, res) => {
    try {
        const response = await axios.get('http://api:5000/ovrednotenje2');
        res.send(response.data); // Spremenjeno, da pošilja celoten odgovor
    } catch (error) {
        res.status(500).send('Napaka pri pridobivanju podatkov');
    }
});


// API poti
app.get('/api/model1', async (req, res) => {
    try {
        const response = await axios.get('http://api:5000/predict/1');
        res.send(response.data.prediction);
    } catch (error) {
        res.status(500).send('Napaka pri pridobivanju podatkov');
    }
});

app.get('/api/model2', async (req, res) => {
    try {
        const response = await axios.get('http://api:5000/predict/2');
        res.send(response.data.prediction);
    } catch (error) {
        res.status(500).send('Napaka pri pridobivanju podatkov');
    }
});

app.listen(port, () => {
    console.log(`App listening at http://localhost:${port}`);
});

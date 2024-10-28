const express = require('express');
const { exec } = require('child_process');
const path = require('path');
const cors = require('cors');
require('dotenv').config(); // Load environment variables from .env file

const app = express();
const port = process.env.PORT || 5000; // Use PORT from .env or default to 5000

// Enable CORS
app.use(cors());

// Serve static files (forecast plot image and CSV files)
app.use('/images', express.static(path.join(__dirname, 'images')));
app.use('/csv', express.static(path.join(__dirname, 'csv')));

// Route to trigger forecasting and generate files
app.get('/api/forecast', (req, res) => {
  exec(
    'set PYTHONIOENCODING=utf-8 && python forecast.py', 
    { encoding: 'utf8', shell: true },
    (error, stdout, stderr) => {
      if (error) {
        console.error(`Error: ${stderr}`);
        res.status(500).json({ error: 'Failed to generate forecast' });
        return;
      }

      // Assuming the Python script generates files at the specified locations
      res.json({
        imageUrl: '/images/forecast.png',
        comparisonCsvUrl: '/csv/comparison_results.csv',
        futurePredictionCsvUrl: '/csv/future_predictions.csv'
      });
    }
  );
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});















// const express = require('express');
// const { exec } = require('child_process');
// const path = require('path');
// const app = express();
// const port = 5000;
// const cors = require('cors');

// // Serve the plot image
// app.use('/images', express.static(path.join(__dirname, 'images')));
// app.use(cors());

// // Route to trigger forecasting
// app.get('/api/forecast', (req, res) => {
//   // Set the PYTHONIOENCODING environment variable for Windows and run the Python script
//   exec('set PYTHONIOENCODING=utf-8 && python forecast.py', { encoding: 'utf8', shell: true }, (error, stdout, stderr) => {
//     if (error) {
//       console.error(`Error: ${stderr}`);
//       res.status(500).json({ error: 'Failed to generate forecast' });
//       return;
//     }
//     res.json({ imageUrl: '/images/forecast.png' });
//   });
// });

// // Start the server
// app.listen(port, () => {
//   console.log(`Server running on port ${port}`);
// });
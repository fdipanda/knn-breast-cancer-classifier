using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class dataPoints
{
    public double[] Entries { get; set; }
    public int Label { get; set; }

    public dataPoints(double[] entry, int label)
    {
        Entries = entry;
        Label = label;
    }
}

public class KNNClassifier
{
    private List<dataPoints> trainingData = new List<dataPoints>();

    public void Training(List<dataPoints> trainingSet)
    {
        trainingData = trainingSet;
    }

    public int Estimate(double[] testEntries, int k)
    {
        var distance = new List<(double Distance, int Label)>();

        foreach (var point in trainingData)
        {
            double dist = GetDistance(point.Entries, testEntries);
            distance.Add((dist, point.Label));
        }

        distance.Sort((a, b) => a.Distance.CompareTo(b.Distance));

        List<(double Distance, int Label)> nearestNeighbors = new List<(double, int)>();
        for (int i = 0; i < k; i++)
        {
            nearestNeighbors.Add(distance[i]);
        }

        int vote = 0;
        foreach (var neighbor in nearestNeighbors)
        {
            vote += neighbor.Label;
        }

        if (vote >= 0)
            return 1;
        else
            return -1;

    }

    public double Analyze(List<dataPoints> testSet, int k, out int[,] confusionMatrix)
    {
        int correct = 0;
        confusionMatrix = new int[2, 2]; 

        foreach (var data in testSet)
        {
            int prediction = Estimate(data.Entries, k);
            if (prediction == data.Label) correct++;

            int predectionIndex;
            if (prediction == 1)
            {
                predectionIndex = 0;
            }
            else
            {
                predectionIndex = 1;
            }

            int actualIndex;
            if (data.Label == 1)
            {
                actualIndex = 0;
            }
            else
            {
                actualIndex = 1;
            }
            confusionMatrix[predectionIndex, actualIndex]++;
        }

        return (double)correct / testSet.Count;
    }

    private double GetDistance(double[] x, double[] y)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double diff = x[i] - y[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    public static List<dataPoints> Normalize(List<dataPoints> dataset)
    {
        int entryCount = dataset[0].Entries.Length;
        double[] min = new double[entryCount];
        double[] max = new double[entryCount];

        for (int i = 0; i < entryCount; i++)
        {
            double minValue = dataset[0].Entries[i];
            double maxValue = dataset[0].Entries[i];

            foreach (var dp in dataset)
            {
                if (dp.Entries[i] < minValue)
                {
                    minValue = dp.Entries[i];
                }

                if (dp.Entries[i] > maxValue)
                {
                    maxValue = dp.Entries[i];
                }
            }

            min[i] = minValue;
            max[i] = maxValue;

        }

        var normalized = new List<dataPoints>();
        foreach (var dp in dataset)
        {
            double[] features = new double[entryCount];
            for (int i = 0; i < entryCount; i++)
            {
                features[i] = (dp.Entries[i] - min[i]) / (max[i] - min[i]);
            }
            normalized.Add(new dataPoints(features, dp.Label));
        }

        return normalized;
    }
}



class Program
{
    static void Main()
    {
        string CSVfile = "wdbc.data.mb.csv";
        var loadData = LoadData(CSVfile);
        loadData = KNNClassifier.Normalize(loadData);

        
        Random rnd = new Random();
        for (int i = loadData.Count - 1; i > 0; i--)
        {
            int j = rnd.Next(i + 1); 
            var temp = loadData[i];
            loadData[i] = loadData[j];
            loadData[j] = temp;
        }


        int trainingSize = (int)(loadData.Count * 0.7);
        var trainingSet = loadData.Take(trainingSize).ToList();
        var testSet = loadData.Skip(trainingSize).ToList();

        var knn = new KNNClassifier();
        knn.Training(trainingSet);

        int[] kValues = { 1, 3, 5, 7, 9 };

        foreach (int k in kValues)
        {
            Console.WriteLine($"k = {k}");
            double accuracy = knn.Analyze(testSet, k, out int[,] confusionMatrix);
            Console.WriteLine($"Accuracy: {accuracy:P2}");

            Console.WriteLine("Confusion Matrix:");
            Console.WriteLine($"        Predicted 1   Predicted -1");
            Console.WriteLine($"Actual 1     {confusionMatrix[0, 0]}             {confusionMatrix[1, 0]}");
            Console.WriteLine($"Actual -1    {confusionMatrix[0, 1]}             {confusionMatrix[1, 1]}");
            Console.WriteLine();
        }
    }

    static List<dataPoints> LoadData(string path)
    {
        var list = new List<dataPoints>();
        foreach (var line in File.ReadLines(path))
        {
            var parts = line.Split(',').Select(double.Parse).ToArray();
            var features = parts.Take(30).ToArray();
            int label = (int)parts[30];
            list.Add(new dataPoints(features, label));
        }
        return list;
    }
}

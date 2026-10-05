// Reads the run CSV and writes the figures as SVG. A chart has to come from the
// data behind it, so the generator lives next to the training run that writes the CSV,
// and the two cannot drift apart.

using System.Globalization;
using System.Text;

namespace Grokking;

public sealed class Row
{
    public double Step, TrainLoss, TrainAcc, TestLoss, TestAcc, ParamSize;
}

internal sealed class Series
{
    public required string Label;
    public required string Color;
    public required List<(double X, double Y)> Points;
    public int Axis; // 0 = left, 1 = right
}

internal sealed class RightAxis
{
    public required string Label;
    public required double Min;
    public required double Max;
    public required double[] Ticks;
    public required string Color;
}

public static class Plot
{
    private const string AxisColor = "#94a3b8";
    private const string TextColor = "#64748b";
    private const string GridColor = "#e2e8f0";

    public static List<Row> ReadCsv(string path)
    {
        List<Row> rows = [];
        foreach (string line in File.ReadLines(path).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] parts = line.Split(',');
            if (parts.Length < 6) continue;
            rows.Add(new Row
            {
                Step = Num(parts[0]),
                TrainLoss = Num(parts[1]),
                TrainAcc = Num(parts[2]),
                TestLoss = Num(parts[3]),
                TestAcc = Num(parts[4]),
                ParamSize = Num(parts[5]),
            });
        }
        return rows;
    }

    private static double Num(string text) =>
        double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);

    private static string F(double value, int decimals) =>
        value.ToString("F" + decimals, CultureInfo.InvariantCulture);

    public static void WriteFigures(string csvPath, string directory)
    {
        List<Row> rows = ReadCsv(csvPath);
        if (rows.Count == 0) return;
        Directory.CreateDirectory(directory);
        double last = rows[^1].Step;

        // 1. The cliff: accuracy on train and on unseen pairs.
        string accuracy = Chart(
            "Generalization happens after memorization",
            "training step",
            "exact-match accuracy",
            last, -0.02, 1.05,
            [0.0, 0.25, 0.5, 0.75, 1.0],
            v => F(v * 100.0, 0) + "%",
            [
                new Series { Label = "train pairs", Color = "#2563eb", Points = rows.Select(r => (r.Step, r.TrainAcc)).ToList() },
                new Series { Label = "unseen pairs", Color = "#dc2626", Points = rows.Select(r => (r.Step, r.TestAcc)).ToList() },
            ],
            null,
            rows.FirstOrDefault(r => r.TestAcc >= 0.5)?.Step ?? last * 0.15);
        File.WriteAllText(Path.Combine(directory, "grokking-cliff.svg"), accuracy);

        // 2. What is happening underneath: losses, log scale.
        static double Log10(double v) => Math.Log10(Math.Max(v, 1e-4));
        string loss = Chart(
            "The train loss reaches zero while the test loss stays flat",
            "training step",
            "cross-entropy loss (log scale)",
            last, Log10(0.01), Log10(10.0),
            [0.01, 0.1, 1.0, 10.0],
            v => v >= 1.0 ? F(v, 0) : F(v, 2),
            [
                new Series { Label = "train", Color = "#2563eb", Points = rows.Select(r => (r.Step, Log10(r.TrainLoss))).ToList() },
                new Series { Label = "unseen", Color = "#dc2626", Points = rows.Select(r => (r.Step, Log10(r.TestLoss))).ToList() },
            ],
            null,
            null);
        File.WriteAllText(Path.Combine(directory, "grokking-loss.svg"), loss);

        // 3. The pressure: the size of the parameters and the unseen accuracy,
        //    each on its own axis.
        double maxSize = rows.Max(r => r.ParamSize) * 1.15;
        string size = Chart(
            "Weight decay shrinks the parameters as the pattern appears",
            "training step",
            "size of the parameters",
            last, 0.0, maxSize,
            [0.0, maxSize * 0.25, maxSize * 0.5, maxSize * 0.75, maxSize],
            v => F(v, 1),
            [
                new Series { Label = "parameter size", Color = "#2563eb", Points = rows.Select(r => (r.Step, r.ParamSize)).ToList() },
                new Series { Label = "unseen accuracy", Color = "#dc2626", Points = rows.Select(r => (r.Step, r.TestAcc)).ToList(), Axis = 1 },
            ],
            new RightAxis
            {
                Label = "accuracy on unseen pairs",
                Min = 0.0,
                Max = 1.0,
                Ticks = [0.0, 0.25, 0.5, 0.75, 1.0],
                Color = "#dc2626",
            },
            null);
        File.WriteAllText(Path.Combine(directory, "grokking-norm.svg"), size);
    }

    /// <summary>
    /// One bar per answer the model could give, from the CSV that --explain writes.
    ///
    /// A model does not answer a question, it spreads a chance over the options.
    /// The bar for the correct answer is the number in the article, and the whole
    /// picture is what changes between the memorized model and the model that
    /// found the rule. The vertical axis is fixed at 0 to 1 so that two of these
    /// figures can be read against each other.
    /// </summary>
    public static void WriteDistribution(string csvPath, string outPath, string title, int answer)
    {
        List<(int Token, double Probability)> bars = [];
        foreach (string line in File.ReadLines(csvPath).Skip(1))
        {
            string[] parts = line.Split(',');
            if (parts.Length >= 2 && int.TryParse(parts[0], out int token))
                bars.Add((token, Num(parts[1])));
        }
        if (bars.Count == 0) return;

        const double width = 900.0, height = 460.0;
        const double marginLeft = 78.0, marginTop = 46.0, marginRight = 24.0, marginBottom = 62.0;
        double plotWidth = width - marginLeft - marginRight;
        double plotHeight = height - marginTop - marginBottom;
        double barWidth = plotWidth / bars.Count;
        double baseline = marginTop + plotHeight;
        double Sy(double y) => marginTop + plotHeight * (1.0 - Math.Clamp(y, 0.0, 1.0));

        StringBuilder svg = new();
        svg.Append($"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{F(width, 0)}\" height=\"{F(height, 0)}\" viewBox=\"0 0 {F(width, 0)} {F(height, 0)}\" font-family=\"ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial,sans-serif\">\n");
        svg.Append($"<text x=\"{F(marginLeft, 0)}\" y=\"26\" font-size=\"17\" font-weight=\"600\" fill=\"{TextColor}\">{title}</text>\n");

        foreach (double tick in new[] { 0.0, 0.25, 0.5, 0.75, 1.0 })
        {
            double y = Sy(tick);
            svg.Append($"<line x1=\"{F(marginLeft, 1)}\" y1=\"{F(y, 1)}\" x2=\"{F(marginLeft + plotWidth, 1)}\" y2=\"{F(y, 1)}\" stroke=\"{GridColor}\" stroke-width=\"1\"/>\n");
            svg.Append($"<text x=\"{F(marginLeft - 10.0, 1)}\" y=\"{F(y + 4.5, 1)}\" font-size=\"13\" fill=\"{TextColor}\" text-anchor=\"end\">{F(tick * 100.0, 0)}%</text>\n");
        }

        (int Token, double Probability) tallest = bars.MaxBy(bar => bar.Probability);
        foreach ((int token, double probability) in bars)
        {
            double x = marginLeft + token * barWidth;
            double y = Sy(probability);
            string colour = token == answer ? "#2563eb" : "#cbd5e1";
            svg.Append($"<rect x=\"{F(x + 1.0, 1)}\" y=\"{F(y, 1)}\" width=\"{F(barWidth - 2.0, 1)}\" height=\"{F(baseline - y, 1)}\" fill=\"{colour}\"/>\n");
        }

        // The tallest bar gets its number, because that is the answer the model gives.
        double tallestX = marginLeft + tallest.Token * barWidth + barWidth / 2.0;
        svg.Append($"<text x=\"{F(tallestX, 1)}\" y=\"{F(Sy(tallest.Probability) - 8.0, 1)}\" font-size=\"13\" font-weight=\"600\" fill=\"#0f172a\" text-anchor=\"middle\">{F(tallest.Probability * 100.0, 0)}%</text>\n");

        svg.Append($"<line x1=\"{F(marginLeft, 1)}\" y1=\"{F(baseline, 1)}\" x2=\"{F(marginLeft + plotWidth, 1)}\" y2=\"{F(baseline, 1)}\" stroke=\"{AxisColor}\" stroke-width=\"1.5\"/>\n");
        for (int token = 0; token < bars.Count; token += 10)
        {
            double x = marginLeft + token * barWidth + barWidth / 2.0;
            svg.Append($"<text x=\"{F(x, 1)}\" y=\"{F(baseline + 20.0, 1)}\" font-size=\"13\" fill=\"{TextColor}\" text-anchor=\"middle\">{token}</text>\n");
        }
        svg.Append($"<text x=\"{F(marginLeft + plotWidth / 2.0, 1)}\" y=\"{F(height - 12.0, 1)}\" font-size=\"13\" fill=\"{TextColor}\" text-anchor=\"middle\">the answer the model is considering</text>\n");

        double legendY = marginTop + 14.0;
        svg.Append($"<rect x=\"{F(marginLeft + plotWidth - 150.0, 1)}\" y=\"{F(legendY - 9.0, 1)}\" width=\"12\" height=\"12\" fill=\"#2563eb\"/>\n");
        svg.Append($"<text x=\"{F(marginLeft + plotWidth - 132.0, 1)}\" y=\"{F(legendY, 1)}\" font-size=\"13\" fill=\"{TextColor}\">the right answer, {answer}</text>\n");

        svg.Append("</svg>\n");
        File.WriteAllText(outPath, svg.ToString());
    }

    private static string Chart(
        string title,
        string xLabel,
        string yLabel,
        double xMax,
        double yMin,
        double yMax,
        double[] yTicks,
        Func<double, string> yFormat,
        List<Series> series,
        RightAxis? right,
        double? shadeUntil)
    {
        const double width = 900.0, height = 460.0;
        double marginRight = right is null ? 150.0 : 232.0;
        const double marginLeft = 78.0, marginTop = 46.0, marginBottom = 52.0;
        double plotWidth = width - marginLeft - marginRight;
        double plotHeight = height - marginTop - marginBottom;

        double Sx(double x) => marginLeft + plotWidth * Math.Clamp(x / xMax, 0.0, 1.0);
        double Sy(double y) => marginTop + plotHeight * (1.0 - Math.Clamp((y - yMin) / (yMax - yMin), 0.0, 1.0));
        double SyRight(double y, RightAxis axis) =>
            marginTop + plotHeight * (1.0 - Math.Clamp((y - axis.Min) / (axis.Max - axis.Min), 0.0, 1.0));

        StringBuilder svg = new();
        svg.Append("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"460\" viewBox=\"0 0 900 460\" font-family=\"ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial,sans-serif\">\n");
        svg.Append($"<text x=\"{F(marginLeft, 0)}\" y=\"26\" font-size=\"17\" font-weight=\"600\" fill=\"{TextColor}\">{title}</text>\n");

        if (shadeUntil is { } until)
        {
            svg.Append($"<rect x=\"{F(Sx(0), 1)}\" y=\"{F(marginTop, 1)}\" width=\"{F(Sx(until) - Sx(0), 1)}\" height=\"{F(plotHeight, 1)}\" fill=\"#f1f5f9\" opacity=\"0.85\"/>\n");
            // The label goes in the gap between the two highest gridlines, where neither
            // curve passes. Placed near the top of the band it was struck out by the line
            // for 100%, and in the middle of the band it landed next to the line for 50%.
            double labelX = Sx(until / 2.0);
            double labelY = (Sy(yTicks[^1]) + Sy(yTicks[^2])) / 2.0;
            svg.Append($"<text x=\"{F(labelX, 1)}\" y=\"{F(labelY + 4.0, 1)}\" font-size=\"12\" fill=\"{TextColor}\" text-anchor=\"middle\">memorization</text>\n");
        }

        foreach (double tick in yTicks)
        {
            double y = Sy(tick);
            svg.Append($"<line x1=\"{F(marginLeft, 1)}\" y1=\"{F(y, 1)}\" x2=\"{F(marginLeft + plotWidth, 1)}\" y2=\"{F(y, 1)}\" stroke=\"{GridColor}\" stroke-width=\"1\"/>\n");
            svg.Append($"<text x=\"{F(marginLeft - 10.0, 1)}\" y=\"{F(y + 4.5, 1)}\" font-size=\"13\" fill=\"{TextColor}\" text-anchor=\"end\">{yFormat(tick)}</text>\n");
        }
        for (int k = 0; k <= 5; k++)
        {
            double value = xMax * k / 5.0;
            double x = Sx(value);
            svg.Append($"<line x1=\"{F(x, 1)}\" y1=\"{F(marginTop, 1)}\" x2=\"{F(x, 1)}\" y2=\"{F(marginTop + plotHeight, 1)}\" stroke=\"{GridColor}\" stroke-width=\"1\"/>\n");
            svg.Append($"<text x=\"{F(x, 1)}\" y=\"{F(marginTop + plotHeight + 20.0, 1)}\" font-size=\"13\" fill=\"{TextColor}\" text-anchor=\"middle\">{F(value, 0)}</text>\n");
        }
        svg.Append($"<line x1=\"{F(marginLeft, 1)}\" y1=\"{F(marginTop + plotHeight, 1)}\" x2=\"{F(marginLeft + plotWidth, 1)}\" y2=\"{F(marginTop + plotHeight, 1)}\" stroke=\"{AxisColor}\" stroke-width=\"1.5\"/>\n");
        svg.Append($"<line x1=\"{F(marginLeft, 1)}\" y1=\"{F(marginTop, 1)}\" x2=\"{F(marginLeft, 1)}\" y2=\"{F(marginTop + plotHeight, 1)}\" stroke=\"{AxisColor}\" stroke-width=\"1.5\"/>\n");
        svg.Append($"<text x=\"{F(marginLeft + plotWidth / 2.0, 1)}\" y=\"{F(height - 12.0, 1)}\" font-size=\"13\" fill=\"{TextColor}\" text-anchor=\"middle\">{xLabel}</text>\n");
        svg.Append($"<text x=\"18\" y=\"{F(marginTop + plotHeight / 2.0, 1)}\" font-size=\"13\" fill=\"{TextColor}\" transform=\"rotate(-90 18 {F(marginTop + plotHeight / 2.0, 1)})\" text-anchor=\"middle\">{yLabel}</text>\n");

        if (right is not null)
        {
            double xr = marginLeft + plotWidth;
            svg.Append($"<line x1=\"{F(xr, 1)}\" y1=\"{F(marginTop, 1)}\" x2=\"{F(xr, 1)}\" y2=\"{F(marginTop + plotHeight, 1)}\" stroke=\"{right.Color}\" stroke-width=\"1.5\"/>\n");
            foreach (double tick in right.Ticks)
            {
                double y = SyRight(tick, right);
                svg.Append($"<line x1=\"{F(xr, 1)}\" y1=\"{F(y, 1)}\" x2=\"{F(xr + 5.0, 1)}\" y2=\"{F(y, 1)}\" stroke=\"{right.Color}\" stroke-width=\"1.5\"/>\n");
                svg.Append($"<text x=\"{F(xr + 10.0, 1)}\" y=\"{F(y + 4.5, 1)}\" font-size=\"13\" fill=\"{right.Color}\">{F(tick * 100.0, 0)}%</text>\n");
            }
            double labelX = xr + 54.0;
            svg.Append($"<text x=\"{F(labelX, 1)}\" y=\"{F(marginTop + plotHeight / 2.0, 1)}\" font-size=\"13\" fill=\"{right.Color}\" text-anchor=\"middle\" transform=\"rotate(-90 {F(labelX, 1)} {F(marginTop + plotHeight / 2.0, 1)})\">{right.Label}</text>\n");
        }

        for (int i = 0; i < series.Count; i++)
        {
            Series s = series[i];
            IEnumerable<string> points = s.Points.Select(p =>
            {
                double y = right is not null && s.Axis == 1 ? SyRight(p.Y, right) : Sy(p.Y);
                return $"{F(Sx(p.X), 1)},{F(y, 1)}";
            });
            svg.Append($"<polyline fill=\"none\" stroke=\"{s.Color}\" stroke-width=\"2.6\" stroke-linejoin=\"round\" points=\"{string.Join(' ', points)}\"/>\n");
            double ly = marginTop + 14.0 + i * 20.0;
            double lx = marginLeft + plotWidth + (right is null ? 16.0 : 70.0);
            svg.Append($"<line x1=\"{F(lx, 1)}\" y1=\"{F(ly, 1)}\" x2=\"{F(lx + 24.0, 1)}\" y2=\"{F(ly, 1)}\" stroke=\"{s.Color}\" stroke-width=\"2.6\"/>\n");
            svg.Append($"<text x=\"{F(lx + 30.0, 1)}\" y=\"{F(ly + 4.5, 1)}\" font-size=\"13\" fill=\"{TextColor}\">{s.Label}</text>\n");
        }

        svg.Append("</svg>\n");
        return svg.ToString();
    }
}

// Grokking in a tiny transformer, on a framework.
//
//   dotnet run -c Release                       # train, log the CSV, write the figures
//   dotnet run -c Release -- --save             # the same, and keep the model
//   dotnet run -c Release -- --infer 12+35      # ask the saved model
//
// This is the same experiment as https://github.com/jacano/grokking-csharp, with
// the whole engine replaced by a framework. There is no node list, no backward
// pass and no derivative here: TorchSharp carries the autograd, the layers and the
// optimizer, and this file only says WHAT the model is and HOW it is trained.
//
// A framework carries the arithmetic: there is no node list, no backward pass and
// no derivative written here. What is left is the model and the training loop.

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;

namespace Grokking;

internal static class Program
{
    // -----------------------------------------------------------------------
    // Settings. The article uses exactly these values.
    // -----------------------------------------------------------------------
    private const int P = 53;             // the modulus: answers go from 0 to 52
    private const double TrainFrac = 0.3; // 30% of the pairs train the model
    private const int NEmbd = 64;         // width of the vector at each position
    private const int NHead = 8;          // attention heads, run in parallel
    private const int Block = 16;         // longest document the model accepts
    private const int Batch = 128;        // pairs read per training step
    private const double Lr = 0.01;       // learning rate
    private const double Wd = 0.0012;     // weight decay. Small on purpose: it is what picks the rule over the table.
    private const int EvalTrain = 512;    // training pairs used for the logged train loss
    private const int EvalBatch = 512;    // documents per evaluation chunk
    private const int Steps = 12000;      // training steps
    private const int EvalEvery = 250;    // one logged row every N steps
    private const int Seed = 42;

    private const string CsvPath = "runs/grokking.csv";
    private const string FigDir = "figures";
    private const string DataDir = "data";
    private const string SavePath = "model.pt"; // written by --save, not part of the repo
    private const string ProbabilityCsv = "runs/probabilities.csv"; // the 53 answers for one sum

    private const int Plus = P;               // token id of '+'
    private const int Eq = P + 1;              // token id of '='
    private const int Bos = P + 2;            // token id of the start token
    private const int Vocab = P + 3;          // 56 tokens in total
    private const int AnswerPos = 4;          // the position whose target is the answer

    private static void Main(string[] args)
    {
        long p = (long)Number(args, "p", P);
        int steps = (int)Number(args, "steps", Steps);
        double wd = Number(args, "wd", Wd);
        double lr = Number(args, "lr", Lr);
        int seed = (int)Number(args, "seed", Seed);
        int evalEvery = (int)Number(args, "eval-every", EvalEvery);
        string infer = Value(args, "infer", "");
        string explain = Value(args, "explain", "");
        string note = Value(args, "note", "");
        bool save = Has(args, "save");

        torch.manual_seed(seed); // the framework brings its own random numbers
        var started = DateTime.UtcNow;

        // ---- the task ----------------------------------------------------
        var (all, cut) = Split(p);
        var trainPairs = all.Take(cut).ToList();
        var testPairs = all.Skip(cut).ToList();
        Tensor trainDocs = Documents(trainPairs, p);
        Tensor testDocs = Documents(testPairs, p);
        WriteRawData(trainPairs, testPairs, p);
        Console.WriteLine(
            $"task: modular addition mod {p} | {p * p} pairs | train {trainPairs.Count} | test {testPairs.Count} | raw copies in {DataDir}/");

        // ---- the model ---------------------------------------------------
        var model = new TinyGpt(Vocab, NEmbd, NHead, Block);
        // The framework brings its own default initialisation: N(0,1) for an
        // embedding, a uniform range for a linear layer. The article uses the one
        // from microgpt, so that the run starts from the scale the article measures.
        using (torch.no_grad())
            foreach ((string _, Parameter parameter) in model.named_parameters())
                parameter.normal_(0, 0.08);
        long nParams = model.parameters().Sum(parameter => parameter.numel());
        Console.WriteLine($"model: {nParams} params | embd {NEmbd} | layers 1 | heads {NHead} | context {Block}");

        // ---- inference only ----------------------------------------------
        if (infer.Length > 0)
        {
            if (!File.Exists(SavePath))
            {
                Console.WriteLine($"no {SavePath} yet: train with --save first");
                return;
            }
            model.load(SavePath); // one line, and the trained weights are back
            Ask(model, infer, p);
            return;
        }

        // ---- one sum, explained -------------------------------------------
        if (explain.Length > 0)
        {
            if (!File.Exists(SavePath))
            {
                Console.WriteLine($"no {SavePath} yet: train with --save first");
                return;
            }
            model.load(SavePath);
            Explain(model, explain, p, note);
            return;
        }

        // ---- training ----------------------------------------------------
        var lossFn = new CrossEntropyLoss();
        // Adam with the decay added to the gradient, as microgpt does, and microgpt's
        // two betas. AdamW is a different optimiser: see the README.
        using var optimiser = optim.Adam(model.parameters(), lr: lr, beta1: 0.85, beta2: 0.99, weight_decay: wd);
        Directory.CreateDirectory(Path.GetDirectoryName(CsvPath) ?? ".");
        using var csv = new StreamWriter(CsvPath);
        csv.NewLine = "\n"; // LF everywhere, so a run never shows the CSV as modified
        csv.WriteLine("step,train_loss,train_acc,test_loss,test_acc,param_norm");
        csv.Flush();

        // The order of the training pairs. Shuffled once, read as a moving window.
        Tensor order = torch.randperm(trainPairs.Count);

        void LogRow(int step)
        {
            (double trainLoss, double trainAcc) = Evaluate(model, trainDocs, lossFn, EvalTrain);
            (double testLoss, double testAcc) = Evaluate(model, testDocs, lossFn, testPairs.Count);
            double size = ParamSize(model);
            csv.WriteLine($"{step},{trainLoss:F6},{trainAcc:F6},{testLoss:F6},{testAcc:F6},{size:F4}");
            csv.Flush();
            Console.WriteLine(
                $"step {step,6} | train loss {trainLoss:F4} acc {trainAcc * 100,5:F1}% | test loss {testLoss:F4} acc {testAcc * 100,5:F1}% | size {size:F3} | {(DateTime.UtcNow - started).TotalSeconds,5:F0}s");
        }

        LogRow(0);

        for (int step = 0; step < steps; step++)
        {
            using var scope = torch.NewDisposeScope();
            int start = (int)((long)step * Batch % trainPairs.Count);
            Tensor index = order.narrow(0, start, Math.Min(Batch, trainPairs.Count - start));
            Tensor batch = trainDocs.index_select(0, index);
            Tensor inputs = batch.narrow(1, 0, 5);   // [START, a, +, b, =]
            Tensor targets = batch.narrow(1, 1, 5);  // the five next tokens

            optimiser.zero_grad();
            Tensor logits = model.forward(inputs);                                  // [B, 5, vocab]
            Tensor loss = lossFn.forward(logits.reshape(-1, Vocab), targets.reshape(-1));
            loss.backward();          // the whole backward pass: one line
            optimiser.step();         // Adam and the decay, one line

            if ((step + 1) % evalEvery == 0 || step == steps - 1) LogRow(step + 1);
        }

        if (save)
        {
            model.save(SavePath); // the whole state dict, one line
            Console.WriteLine($"saved the model to {SavePath}");
        }

        csv.Dispose();
        Plot.WriteFigures(CsvPath, FigDir);
        Console.WriteLine($"wrote the figures to {FigDir}/");

        foreach ((long a, long b) in new[] { (12L, 35L), (7L, 44L), (50L, 50L), (0L, 13L) })
            Ask(model, $"{a}+{b}", p);
        Console.WriteLine($"total {(DateTime.UtcNow - started).TotalSeconds:F0}s");
    }

    // -----------------------------------------------------------------------

    /// <summary>Every pair, shuffled once. The first 30% trains, the rest tests.</summary>
    private static (List<(long A, long B)> Pairs, int Cut) Split(long p)
    {
        List<(long, long)> pairs = [];
        for (long a = 0; a < p; a++)
            for (long b = 0; b < p; b++)
                pairs.Add((a, b));
        long[] order = torch.randperm(pairs.Count).data<long>().ToArray();
        List<(long A, long B)> shuffled = [.. order.Select(i => pairs[(int)i])];
        return (shuffled, (int)(shuffled.Count * TrainFrac + 0.5));
    }

    /// <summary>One row per pair: [START, a, '+', b, '=', c].</summary>
    private static Tensor Documents(List<(long A, long B)> pairs, long p)
    {
        long[] flat = new long[pairs.Count * 6];
        for (int i = 0; i < pairs.Count; i++)
        {
            (long a, long b) = pairs[i];
            flat[i * 6 + 0] = Bos;
            flat[i * 6 + 1] = a;
            flat[i * 6 + 2] = Plus;
            flat[i * 6 + 3] = b;
            flat[i * 6 + 4] = Eq;
            flat[i * 6 + 5] = (a + b) % p;
        }
        return torch.tensor(flat).reshape(pairs.Count, 6);
    }

    /// <summary>Write both splits as plain text, in the shape the model reads.</summary>
    private static void WriteRawData(List<(long A, long B)> train, List<(long A, long B)> test, long p)
    {
        Directory.CreateDirectory(DataDir);
        // A newline per line, always: the same bytes on Windows, Linux and macOS,
        // so a run never shows the files as modified.
        WriteSplit(Path.Combine(DataDir, "train.txt"), train, p);
        WriteSplit(Path.Combine(DataDir, "test.txt"), test, p);
    }

    private static void WriteSplit(string path, List<(long A, long B)> pairs, long p)
    {
        string text = string.Join('\n', pairs.Select(x => $"{x.A} + {x.B} = {(x.A + x.B) % p}"));
        File.WriteAllText(path, text + "\n");
    }

    /// <summary>Mean loss over the five positions, and exact match on the answer.</summary>
    private static (double Loss, double Acc) Evaluate(torch.nn.Module<Tensor, Tensor> model, Tensor docs, torch.nn.Module<Tensor, Tensor, Tensor> lossFn, int count)
    {
        double totalLoss = 0;
        long hits = 0;
        for (int i = 0; i < count; i += EvalBatch)
        {
            using var scope = torch.NewDisposeScope();
            int size = Math.Min(EvalBatch, count - i);
            Tensor batch = docs.narrow(0, i, size);
            Tensor inputs = batch.narrow(1, 0, 5);
            Tensor targets = batch.narrow(1, 1, 5);
            Tensor logits = model.forward(inputs);
            totalLoss += lossFn.forward(logits.reshape(-1, Vocab), targets.reshape(-1)).item<float>() * size;
            Tensor predicted = logits.argmax(-1).narrow(1, AnswerPos, 1).squeeze(1);
            Tensor wanted = targets.narrow(1, AnswerPos, 1).squeeze(1);
            hits += predicted.eq(wanted).sum().item<long>();
        }
        return (totalLoss / count, hits / (double)count);
    }

    /// <summary>One number for the whole model: how large the parameters are together.</summary>
    private static double ParamSize(torch.nn.Module<Tensor, Tensor> model) =>
        Math.Sqrt((double)model.parameters().Sum(parameter => parameter.detach().pow(2).sum().item<float>()));

    /// <summary>Print the answer for "a+b" and the three most likely ones.</summary>
    private static void Ask(torch.nn.Module<Tensor, Tensor> model, string expression, long p)
    {
        if (!Parse(expression, p, out long a, out long b)) return;
        double[] probs = Distribution(model, a, b, p);
        (double value, long index)[] ranked = probs
            .Select((value, index) => (value, (long)index))
            .OrderByDescending(x => x.value)
            .Take(3)
            .ToArray();
        string best = ranked[0].index == (a + b) % p ? "ok" : "wrong";
        string top = string.Join(", ", ranked.Select(x => $"{x.index} ({x.value * 100:F0}%)"));
        Console.WriteLine($"inference {a}+{b} = {ranked[0].index}  [{best}]  top: {top}");
    }

    /// <summary>
    /// Write every one of the 53 answers the model is considering, and draw them.
    ///
    /// A model does not answer, it spreads a probability over the options. The
    /// figure is that spread, and the number in the article is the height of the
    /// tallest bar.
    /// </summary>
    private static void Explain(torch.nn.Module<Tensor, Tensor> model, string expression, long p, string note)
    {
        if (!Parse(expression, p, out long a, out long b)) return;
        double[] probs = Distribution(model, a, b, p);

        Directory.CreateDirectory(Path.GetDirectoryName(ProbabilityCsv) ?? ".");
        using (StreamWriter writer = new(ProbabilityCsv))
        {
            writer.NewLine = "\n";
            writer.WriteLine("token,probability");
            for (int token = 0; token < probs.Length; token++)
                writer.WriteLine($"{token},{probs[token]:F6}");
        }

        int answer = (int)((a + b) % p);
        Plot.WriteDistribution(
            ProbabilityCsv,
            Path.Combine("figures", "grokking-probabilities.svg"),
            $"The chance the model gives to each answer, for {a} + {b}" + (note.Length > 0 ? $" — {note}" : ""),
            answer);
        Console.WriteLine($"wrote {ProbabilityCsv} and figures/grokking-probabilities.svg");
        Console.WriteLine($"the answer is {answer}, and the model gives it {probs[answer] * 100:F1}%");
    }

    /// <summary>One probability per answer token, the way the model spreads them.</summary>
    private static double[] Distribution(torch.nn.Module<Tensor, Tensor> model, long a, long b, long p)
    {
        using var scope = torch.NewDisposeScope();
        Tensor prompt = torch.tensor(new long[] { Bos, a, Plus, b, Eq }).reshape(1, 5);
        Tensor logits = model.forward(prompt).squeeze(0).narrow(0, AnswerPos, 1).squeeze(0); // [vocab]
        Tensor probs = logits.narrow(0, 0, (int)p).softmax(0); // the p number tokens
        float[] values = probs.data<float>().ToArray();
        return [.. values.Select(value => (double)value)];
    }

    private static bool Parse(string expression, long p, out long a, out long b)
    {
        string[] parts = expression.Split('+');
        if (parts.Length != 2 || !long.TryParse(parts[0].Trim(), out long first) || !long.TryParse(parts[1].Trim(), out long second))
        {
            Console.WriteLine($"expected a+b, got {expression}");
            a = 0;
            b = 0;
            return false;
        }
        a = ((first % p) + p) % p;
        b = ((second % p) + p) % p;
        return true;
    }

    private static string Arg(string[] args, string name)
    {
        string key = "--" + name;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == key && i + 1 < args.Length) return args[i + 1];
            if (args[i].StartsWith(key + "=", StringComparison.Ordinal)) return args[i][(key.Length + 1)..];
        }
        return "";
    }

    private static string Value(string[] args, string name, string fallback) =>
        Arg(args, name) is { Length: > 0 } found ? found : fallback;

    private static double Number(string[] args, string name, double fallback) =>
        Arg(args, name) is { Length: > 0 } text ? double.Parse(text, System.Globalization.CultureInfo.InvariantCulture) : fallback;

    private static bool Has(string[] args, string name) => args.Contains("--" + name);
}

/// <summary>
/// The whole architecture. Four layers of the framework and the two lines that
/// wire them, and nothing else: no node, no gradient, no backward pass.
/// </summary>
internal sealed class TinyGpt : torch.nn.Module<Tensor, Tensor>
{
    private readonly Embedding _wte;
    private readonly Embedding _wpe;
    private readonly Linear _wq;
    private readonly Linear _wk;
    private readonly Linear _wv;
    private readonly Linear _wo;
    private readonly Linear _fc1;
    private readonly Linear _fc2;
    private readonly Linear _lmHead;
    private readonly int _embd;
    private readonly int _headDim;
    private readonly int _heads;

    public TinyGpt(int vocab, int embd, int heads, int block)
        : base(nameof(TinyGpt))
    {
        _embd = embd;
        _heads = heads;
        _headDim = embd / heads;
        _wte = torch.nn.Embedding(vocab, embd);
        _wpe = torch.nn.Embedding(block, embd);
        _wq = torch.nn.Linear(embd, embd, hasBias: false);
        _wk = torch.nn.Linear(embd, embd, hasBias: false);
        _wv = torch.nn.Linear(embd, embd, hasBias: false);
        _wo = torch.nn.Linear(embd, embd, hasBias: false);
        _fc1 = torch.nn.Linear(embd, 4 * embd, hasBias: false);
        _fc2 = torch.nn.Linear(4 * embd, embd, hasBias: false);
        _lmHead = torch.nn.Linear(embd, vocab, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor index)
    {
        long batch = index.shape[0];
        long length = index.shape[1];
        Tensor positions = arange(length, dtype: ScalarType.Int64, device: index.device);
        Tensor x = _wte.forward(index) + _wpe.forward(positions); // [B, T, C]

        // --- attention: the framework knows how to do this, and how to mask it ---
        Tensor h = RmsNorm(x);
        Tensor q = _wq.forward(h).reshape(batch, length, _heads, _headDim).transpose(1, 2);
        Tensor k = _wk.forward(h).reshape(batch, length, _heads, _headDim).transpose(1, 2);
        Tensor v = _wv.forward(h).reshape(batch, length, _heads, _headDim).transpose(1, 2);
        // The mask that keeps a position from reading the future is the library's job.
        // (TorchSharp spells the argument 'is_casual'.)
        Tensor attended = scaled_dot_product_attention(q, k, v, is_casual: true);
        x = x + _wo.forward(attended.transpose(1, 2).reshape(batch, length, _embd));

        // --- MLP: project up, run the non-linearity, project back ---
        x = x + _fc2.forward(relu(_fc1.forward(RmsNorm(x))));
        return _lmHead.forward(x); // [B, T, vocab]
    }

    /// <summary>Keep every vector in a stable range. No parameters, as in microgpt.</summary>
    private static Tensor RmsNorm(Tensor x) =>
        x / sqrt(x.pow(2).mean(new long[] { -1 }, keepdim: true) + 1e-5);
}

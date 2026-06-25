// Plain-English definitions for ML beginners — 2-3 sentences, no jargon.
export const GLOSSARY = {
  gradient_heatmap: {
    title: "Gradient heatmap",
    text: "Colors each neuron by how strongly it's currently learning (its “gradient”). Bright neurons are making big adjustments this step; dim ones are barely changing. It's a way to see where learning is actually happening inside the network.",
  },
  dead_neurons: {
    title: "Dead neurons",
    text: "A neuron that has stopped responding — it outputs zero for almost every image and no longer learns. This usually happens when the learning rate is too high. Lots of dead neurons means the network is wasting part of its brain.",
  },
  epoch: {
    title: "Epoch",
    text: "One full pass of all the training images through the network. After each epoch the network has seen every example once and nudged its settings to do a little better. More epochs means more practice.",
  },
  loss: {
    title: "Loss",
    text: "A single number measuring how wrong the network's guesses are right now. Lower is better — zero would mean perfect. Training is just the process of pushing this number down.",
  },
  batch_size: {
    title: "Batch size",
    text: "How many images the network looks at before it updates its settings once. Smaller batches update more often but more erratically; larger batches are smoother but slower to react.",
  },
  learning_rate: {
    title: "Learning rate",
    text: "How big a step the network takes when it adjusts itself after each batch. Too small and it learns painfully slowly; too large and it overshoots and can break (dead neurons). It's the single most important dial to tune.",
  },
  data_fraction: {
    title: "Data fraction",
    text: "What slice of the full 60,000 training images to actually use. A smaller fraction trains faster but the network sees less variety. 0.1 means it trains on 10% of the images.",
  },
  train_acc: {
    title: "Train accuracy",
    text: "The percentage of training images the network currently gets right. These are images it has already studied, so this number tends to look optimistic.",
  },
  test_acc: {
    title: "Test accuracy",
    text: "The percentage of unseen images the network gets right — ones it never trained on. This is the honest measure of whether it truly learned to read digits, rather than just memorizing the examples it saw.",
  },
  optimizer: {
    title: "Optimizer",
    text: "The recipe the network uses to turn gradients into weight updates. SGD is the classic approach — steady but slow. Adam is smarter about adjusting each weight individually and usually trains faster.",
  },
  backpropagation: {
    title: "Backpropagation",
    text: "The algorithm that lets the network learn from mistakes. After each guess it works backwards from the error to figure out how every internal weight should change to be a little less wrong. Repeated over and over, this is how the network improves.",
  },
} as const;

export type GlossaryKey = keyof typeof GLOSSARY;

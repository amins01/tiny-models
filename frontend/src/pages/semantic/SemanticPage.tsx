import { useEffect, useRef, useState } from "react";

import InputCanvas from "../../components/InputCanvas";
import OutputCanvas from "../../components/OutputCanvas";
import { useInference } from "../../hooks/useInference";
import InferenceDashboard from "../../components/InferenceDashboard";
import type { InferenceTelemetry } from "../../types";
import { generateGaussianNoise } from "../../utils/tensorUtils";
import { MODEL_CONFIGS } from "../../constants";
import { tensorToCanvas } from "../../utils/imageUtils";
import Article from "../../components/Article";

import { Typography, Link } from "@mui/material";
import Alert from "@mui/material/Alert";
import Collapse from "@mui/material/Collapse";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import { useTheme } from "@mui/material/styles";

export default function SemanticPage() {
  const MODEL_TYPE = "semantic";
  const CONFIGS = MODEL_CONFIGS[MODEL_TYPE];
  const [outputCanvas, setOutputCanvas] = useState<any>(null);
  const {ready, generate} = useInference(CONFIGS.modelPath, MODEL_TYPE);
  const [noiseData, setNoiseData] = useState<Float32Array>(generateGaussianNoise(CONFIGS.imgDims.reduce((a, b) => a * b, 1)));
  const latestSketchData = useRef<Float32Array | null>(null);
  const isProcessing = useRef<boolean>(false);
  const [openAlert, setOpenAlert] = useState(true);
  const theme = useTheme();

  const [dashboardState, setDashboardState] = useState<InferenceTelemetry>({
    step: 0,
    totalSteps: 10,
    stepTime: 0,
    totalTime: 0,
    image: new Float32Array(0),
    // velocity: new Float32Array(0),
    sketch: new Float32Array(0),
  });

  const drawLatest = async () => {
    if (!latestSketchData.current) {
      isProcessing.current = false
      return;
    }

    isProcessing.current = true
    const dataTensor = latestSketchData.current
    
    setDashboardState(prev => ({
        ...prev,
        step: 0,
        totalTime: 0,
        image: new Float32Array(0)
    }));

    latestSketchData.current = null
    const result = await generate(dataTensor!, dashboardState.totalSteps, noiseData, setDashboardState, null);
    const canvas = tensorToCanvas(new Float32Array(result as Float32Array), MODEL_TYPE, true);
    setOutputCanvas(canvas);
    drawLatest();
  }

  const handleDraw = async (tensorData: Float32Array) => {
    if (!ready) return;

    latestSketchData.current = tensorData

    if (!isProcessing.current) {
      drawLatest()
    }
  };

  useEffect(() => {
    if (dashboardState.sketch?.length > 0) handleDraw(dashboardState.sketch);
  }, [noiseData]);

  // useEffect(() => {
  //   setDashboardState(prev => ({
  //       ...prev,
  //       sketch: new Float32Array(0)
  //   }));
  //   setModelConfigs(MODEL_CONFIGS[modelType]);
  //   setModelPath(MODEL_CONFIGS[modelType].modelPath);
  // }, [modelType]);

  const onNoiseReset = async () => {
    const newNoise = generateGaussianNoise(CONFIGS.imgDims.reduce((a, b) => a * b, 1));
    setNoiseData(newNoise);
  };

  const onStepChange = (newStep: number) => {
    setDashboardState(prev => ({
        ...prev, totalSteps: newStep
    }));
  };

  const BLOG_CONTENT = `
  # Motivation
  Local high quality image generation is out of reach for many consumers. 
  This is partly due to the desire of modern AI labs to build general purpose models, which requires significant model capacity, i.e., more parameters,
  and who says more parameters, says more VRAM requiring hardward accelerators (e.g., GPUs) that are anything but affordable these days.
  However, many use cases involve narrow tasks that can be adequately performed by much smaller models able to run entirely client-side.
  This means: **zero inference-related server costs and more data privacy, since the data never has to leave the user's device**. 

  Small local models have garnered significant interest for text generation but are often overlooked for image generation.
  The goal of this project is to **demonstrate the advantages and capabilities of lightweight image models (trained from scratch) directly running on the client's CPU**. 
  Moreover, it allows me to apply some interesting approaches suggested in different papers to a practical use case.
  
  ## Task / Dataset
  Given the compute constraints and limited model capacity, our training dataset must comprise of images that still look good and coherent at low resolution.
  Moreover, the conditioning should be more interesting than a simple class in order to make the demo more interactive.
  
  One option that comes to mind is text conditioning. 
  While it would offer more flexibility, it may require too much capacity from the model given the vast space of text prompts.
  It would also require the use of a text encoder which should ideally be avoided given our computational constraints.
  
  This leads us to the next option: conditioning the generation on a drawing that would provide the model with rich spatial information.
  *This sounds interesting but where would we find the high quality image/drawing pairs to train our model?*
  The closest that I could find (on kaggle and hugging face) were segmentation datasets, since they contain semantic maps that roughly look like drawings.
  The problem is that most of the segmentation datasets (e.g., [COCO](https://cocodataset.org/#explore)) were built for object recognition and thus contain a large number of "in the wild" unaesthetic low quality images, which are not desirable for image generation (see Figure 1).

  ![Figure 1: Unaesthetic sample from the COCO training set.](/tiny-models/coco_unaesthetic.jpg)

  But there's a solution: **we can generate the segementation maps ourselves using a pretrained segmentation model**. 
  That would allow us to focus on finding a high quality image dataset without worrying about whether it includes segmentation maps. 
  The chosen dataset: [Landscapes HQ](https://www.kaggle.com/datasets/dimensi0n/lhq-1024). 
  It contains 90k high quality landscape images and since landscapes contain a lot of low frequency details, they retain their coherence even at low resolution. 
  *Note: the idea of semantic landscape generation isn't new; after some research I found Nvidia's work ([Park et al., 2019](https://arxiv.org/pdf/1903.07291)) 
  which proposes the same idea but using a generative adversarial network (GAN).*

  ## Data Processing
  I opted for a [SegFormer fine-tuned on ADE20k](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512), which can output 150 classes (from [ADE20k](https://ade20k.csail.mit.edu/)).
  In order to simplify the task for our lightweight model, I mapped the classes predicted by the pretrained SegFormer to 6 general classes: 
  sky, greenery (e.g, grass and trees), sand, rocks (including mountains), water, and undefined.
  We resize the images to the target resolution for our model and feed them to the pretrained SegFormer.
  We then map the output to the 6 aforementioned general classes. **Result: a clean dataset of high quality landscape images each paired with a simplified segmentation map**.

  ![Figure 2: LHQ image along with the segementation map extracted by the pretrained SegFormer.](/tiny-models/lhq_gt.png)

  ## Model Architecture
  I decided to opt for a Diffusion Transformer (DiT) as model architecture since it has shown to work so well for image generation.
  The configuration is very similar to the DiT-S one from the original [DiT paper](https://arxiv.org/pdf/2212.09748) with a few differences:
  1. The model is trained using a [flow matching](https://arxiv.org/pdf/2210.02747) objective rather than standard diffusion.
  2. The original paper integrates the class conditioning by summing its embedding with the timestep's and using adaLN (by passing the resulting tensor to each DiT block's conditioning MLP which outputs tensors used to scale (σ) and shift (μ) the output of certain layers).
  Instead, I opted for a simpler and potentially more effective approach for our task:
  the drawing (i.e., segmentation map) is converted into a 7 dimensional (for the 7 semantic classes from our palette) one hot encoding for each pixel of the input image. 
  This results in a conditioning tensor of shape (batch_size, num_semantic_classes, image_height, image_width).
  The tensor is then concatenated to the input image before patchification. 
  This allows us to retain the spatial information from the drawing for a minimal increase in parameter count (only in the patchify 2D convolutional layer).
  Like the original DiT, the timestep conditioning is still integrated using adaLN-Zero.
  3. Unlike most modern image models, the model is trained in pixel space, to avoid the use of an image encoder and decoder, which would bring their own set of problems for inference (and we don't really need them, see Training section).

  ![Figure 3: From the [DiT paper](https://arxiv.org/pdf/2212.09748). It shows the architecture our model mostly follows.](/tiny-models/dit.png)

  There are a few hyperparameters that must be adjusted for a good balance between image quality and inference latency.
  The user should get quick feedback with every stroke. 
  I'm thus aiming for an inference time of ~1s (for full generation, i.e., all sampling steps combined).
  As shown in the original DiT paper, a smaller patch size leads to better image quality (it's a way to scale compute).
  In order to determine the optimal patch size for our model, we export a randomly intialized version of it, plug it to the frontend, and measure the latency and overall *ux vibes*.
  The patch size offering the best balance between latency and quality (on my CPU) was the one leading to 256 patches (16x16).
  This means that for an image of size 128x128, the patch size (p) should be 8, for 256x256 p=16, and so on.
  
  Here's a simplified version of our model's forward function:
  \`\`\`python
  def forward(self, x, t, cond):
    bs = x.shape[0]

    # patchify x + c
    x_c = torch.cat([x, cond], dim=1)
    patches = self.patch_conv(x_c) # [bs, dim, n_patches_side, n_patches_side]
    patches = patches.permute(0, 2, 3, 1).reshape(bs, -1, self.model_dim) # [bs, n_patches, dim]
    patches += self.pos_emb
    
    # embed t
    cond_emb = self.timestep_emb(t)

    # DiT blocks
    for layer in self.dit_layers:
        patches = layer(patches, cond_emb, key_padding_mask)

    # layer norm
    norm_out = self.norm(patches)
    scale_shift_params = self.adaln_mlp(cond_emb).unsqueeze(1)
    scale, shift = torch.chunk(scale_shift_params, 2, dim=-1)
    x = norm_out * (1 + scale) + shift

    # linear and reshape
    out = self.out_head(x) # [bs, n_patches, p*p*c]
    out = out.reshape(
        bs, self.n_patches_side, self.n_patches_side,
        self.image_channels, self.patch_size, self.patch_size
    ).permute(0, 3, 1, 4, 2, 5)
    out = out.reshape(bs, self.image_channels, self.image_size, self.image_size)

    return out
  \`\`\`

  More architecture details:
  | Parameters | Patches | Layers | Hiddem Dim | Heads |
  | :--- | :--- | :--- | :--- | :--- |
  | ~22 Million | 256 | 12 | 384 | 6 |

  ## Frontend
  The frontend is built in React + Typescript.
  Many of the UI components come from Material UI. 
  The demo allows the user to draw using a color palette that maps to the general segmentation classes,
  after which the ui would show the model prediction using the drawing as conditioning.
  The user can adjust the number of sampling steps and manually reset the initial Gaussian noise, which is fixed throughout generations so that the output image doesn't drastically change.

  Since we want the model to run entirely client-side, I avoided going for the traditional "exposing the model's functions via a Python API" setup.
  Instead, I exported the model to Open Neural Network Exchange (ONNX) format, which saves a computation graph representing the operations performed by the model in a forward pass.
  We can then run inference directly on the client's CPU using ONNX Runtime and WebAssembly (WASM).
  To avoid blocking the main thread, long running jobs (model loading and inference) are ran on a separate thread using web workers.
  Without web workers, the app would be constantly freezing and unusable.

  ## Training
  Before training the model, we must determine the image resolution.
  While image resolution has a marginal impact on the model's parameter count (only adds parameters to the output head), 
  it's bounded by the model's hidden dimension, which has a significant impact on the parameter count.
  In other words, the bigger the hidden dimension is, the higher the image resolution our model can support.
  Since we're replicating the DiT-S configuration, the model's hidden dimension is of size 384.
  After some experiments, the maximum image resolution that the model could learn without significant artifacts is 128x128.

  ![Figure 4: Conditioning and images generated with 10 Euler sampling steps by our DiT trained on 128x128 images (patch size = 8) using standard flow matching (predicting velocity).](/tiny-models/dit_pred_v_loss_v_128.png)

  As you can probably see, the quality of the generated images (Figure 4) isn't great.
  The generated images don't seem to properly follow the conditioning and the low resolution doesn't help.
  That's when I remembered the recent [JiT paper (Back to Basics: Let Denoising Generative Models Denoise)](https://arxiv.org/pdf/2511.13720),
  showing the benefits of training a model to directly predict the image (x) instead of noise or velocity.
  In a nutshell, the [manifold assumption](https://www.molgen.mpg.de/3659531/MITPress--SemiSupervised-Learning.pdf)
  hypothesizes that a lot of high dimensional data, especially natural/real world data, lie on a low dimensional manifold.
  Under that lens, making the model generate a noise quantity (lying in a high dimensional manifold)
  would be suboptimal compared to directly making it generate the image (lying in a low dimensional),
  since the model would waste some of its capacity on retaining high dimensional information unrelated to the image.

  ![Figure 5: From the [JiT paper](https://arxiv.org/pdf/2511.13720). It illustrates the motivation behind x prediciton under the manifold assumption.](/tiny-models/jit_manifold.png)

  Note that, as mentioned in the paper, we can have any combination of prediction and loss (x, ε, v) by applying a transformation to the model's output (Figure 6),
  which also means that during inference, we can iteratively sample from a model trained on any pred-loss combination.

  ![Figure 6: From the [JiT paper](https://arxiv.org/pdf/2511.13720). It shows the possible prediction-loss combinations.](/tiny-models/jit_pred_loss_comb.png)

  For 128x128 images with a patch size of 8, training the model to predict x with velocity loss led to worse results than using velocity prediction and loss.
  This is expected since, as shown in the paper, the negative effects of noise prediction emerge when the patch dimensionality is high relative to the model's hidden dim (near or exceeding), 
  which isn't the case here since the patch dimensionality is 192 (8x8x3) and the model's hidden dim is 384 (double).
  So, in that case, the model has enough capacity for the high dimensional velocity.
  However, when scaling the image resolution to 256x256 with a patch size of 16, 
  training the model on x prediction and velocity loss led to a drastic improvement in generated image quality (Figure 7).
  The same setup (256x256 with patch size of 16) using velocity for prediction and loss led to significantly worse images (with noise artifacts overwhelming the generated images).
  
  ![Figure 7: Conditioning and images generated with 10 Euler sampling steps by our DiT trained on 256x256 images (patch size = 16) 
  to directly predict x **(left)** and velocity **(right)**, both with velocity loss.](/tiny-models/dit_256_pred_loss_comparison.png)
  
  *Note: the experiments that I've done are far from exhaustive. 
  I believe that, with the right optimizations, the supported image size can be greatly improved without sacrificing latency.*
  
  The model was trained for 200k steps with a batch size of 256 on the LHQ images paired with the extracted general segmentation maps (only augmentation is horizontal flip).
  I used AdamW (β1=0.9, β2=0.95) as optimizer with a weight decay coefficient of 0.05 and a constant learning rate of 1e-4.
  An exponential moving average (EMA) of the model weights is kept for inference.

  Here's a simplified version of the training loop:
  \`\`\`python
  while cur_step < args.train_steps:
    cond, img = next(train_dataloader)

    t = torch.rand((per_gpu_bs,), device=device)
    eps = torch.randn_like(img)
    expanded_t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    noisy_img = expanded_t * img + (torch.ones_like(expanded_t) - expanded_t) * eps
    model_out = model(noisy_img, t, cond)

    # prepare target
    if args.loss == "x":
        target = img
    elif args.loss == "v":
        target = img - eps
    
    # prepare pred
    if args.pred == args.loss:
        pred = model_out
    elif args.pred == "x" and args.loss == "v":
        pred = (model_out - noisy_img) / ((torch.ones_like(expanded_t) - expanded_t)).clip(0.05) # vθ=(xθ−zt)/(1−t)
    elif args.pred == "v" and args.loss == "x":
        pred = (torch.ones_like(expanded_t) - expanded_t) * model_out + noisy_img # xθ=(1−t)vθ+zt
    
    loss = torch.nn.functional.mse_loss(pred, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    cur_step += 1
  \`\`\`

  ## TODOs
  There are still many ways to improve this project, namely by:
  - Smoothing the segmentation maps, mimicking real user drawings to limit train-test mismatch
  - Training the model on a larger and more diverse dataset
  - Adding text conditioning
  - Improving the balance between generation quality and latency
  (e.g., by quantizing the model or by using Mean Flows for one step generation (Geng et al., 2025 [1](https://arxiv.org/pdf/2505.13447) and [2](https://arxiv.org/pdf/2512.02012)))

  That's it! Hope I was able to convey the potential of client-side inference with generative image models.
  `;

  return (
    <>
      <div className="column container">
        <Typography variant="h3" fontWeight="400" sx={{ mt: 1, mb: 2, letterSpacing: "-0.02em" }}>
          Semantic Image Generation
        </Typography>
        <Typography variant="body1" component="div"
          sx={{
            fontSize: "1.125rem",
            lineHeight: 1.75,
            textAlign: "left",
            paddingLeft: "1em",
            paddingRight: "1em",
            maxWidth: "1100px"
          }}
        >
          Drawing to landscape image using a diffusion model trained from scratch running fully locally (CPU). Code's available&nbsp;
          <Link
            href="https://github.com/amins01/tiny-models"
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              color: theme.palette.primary.main,
              textDecoration: "underline",
              textUnderlineOffset: "3px",
              fontWeight: 500,
            }}
          >
            here
          </Link>.
        </Typography>
        <Collapse in={openAlert}>
          <Alert 
            severity="warning"
            sx={{ maxWidth: "900px" }}
            action={
              <IconButton
                color="inherit"
                size="small"
                onClick={() => setOpenAlert(false)}
              >
                <CloseIcon fontSize="inherit" />
              </IconButton>
            }
          >
            Inference is slower on the deployed site because it's hosted on GitHub Pages (which I chose for a free "set and forget" setup) 
            and it doesn't allow the browser headers required for WASM multithreading.
            Running it locally should lead to much faster inference.
          </Alert>
        </Collapse>
      </div>
      <div className="row container">
        <div className="column">
          <InputCanvas
            onDraw={handleDraw}
            modelType={MODEL_TYPE}
          />
        </div>
        <div className="column">
          <OutputCanvas canvas={outputCanvas} displayText="Start by drawing a landscape on the left canvas"/>
        </div>
        <div className="column">
          <InferenceDashboard
            currentStep={dashboardState.step}
            totalSteps={dashboardState.totalSteps}
            stepTime={dashboardState.stepTime}
            totalTime={dashboardState.totalTime}
            imageData={dashboardState.image.length > 0 ? dashboardState.image : null}
            // velocityData={dashboardState.velocity.length > 0 ? dashboardState.velocity : null}
            onStepChange={onStepChange}
            onNoiseReset={onNoiseReset}
            isProcessing={isProcessing.current}
            isModelLoaded={ready}
            modelType={MODEL_TYPE}
          />
        </div>
      </div>
      <Article content={BLOG_CONTENT} />
    </>
  );
}
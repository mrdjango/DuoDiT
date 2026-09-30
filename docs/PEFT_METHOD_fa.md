# توضیح روش PEFT استفاده‌شده در DuoDiT

## خلاصه

روش استفاده‌شده در این پروژه یک روش **Parameter-Efficient Fine-Tuning (PEFT)** از نوع **adapter-style partial fine-tuning** است. در این روش، مدل پایه‌ی `facebook/DiT` به عنوان backbone اصلی حفظ می‌شود و بیشتر پارامترهای آن freeze می‌شوند. سپس فقط یک مسیر جانبی قابل آموزش، یعنی مسیر `x2`، به همراه چند لایه‌ی مرتبط با خروجی آموزش داده می‌شود.

بنابراین این روش **LoRA کلاسیک** نیست؛ چون در لایه‌های اصلی transformer ماتریس‌های low-rank تزریق نشده‌اند. روش حاضر بیشتر شبیه یک **auxiliary adapter / side-stream adapter** است که به مدل DiT اضافه شده و با تعداد کمی از پارامترها fine-tune می‌شود.

## ایده‌ی اصلی

ایده‌ی اصلی این است که کیفیت و دانش مدل pretrained DiT حفظ شود، اما مدل بتواند با یک شاخه‌ی کم‌پارامتر جدید به داده‌ها یا کلاس‌های هدف سازگار شود. به جای fine-tune کردن کل مدل DiT، فقط بخش‌هایی آموزش داده می‌شوند که مسئول مسیر `x2` و تطبیق خروجی هستند.

این کار دو هدف دارد:

1. کاهش هزینه‌ی محاسباتی و حافظه نسبت به full fine-tuning.
2. کاهش خطر خراب شدن دانش pretrained مدل اصلی.

## پارامترهای freeze شده

در اسکریپت `train_x2_finetune.py` ابتدا همه‌ی پارامترهای مدل freeze می‌شوند. در نتیجه بخش‌های اصلی زیر train نمی‌شوند:

- `x_embedder`
- `t_embedder`
- `y_embedder`
- بلوک‌های اصلی transformer در DiT، یعنی `blocks`
- بیشتر backbone اصلی pretrained DiT

این یعنی مسیر اصلی denoising در DiT تا حد زیادی همان دانش pretrained خود را حفظ می‌کند.

## پارامترهای قابل آموزش

بعد از freeze کردن کل مدل، فقط این بخش‌ها دوباره قابل آموزش می‌شوند:

- `x2_embedder`
- `x2_cls_tokens`
- `x2_vit_block`
- `x2_vit_proj_in`، در صورت وجود
- `x2_vit_proj_out`، در صورت وجود
- `final_layer`

پس توصیف دقیق روش این است:

> Frozen DiT backbone with a trainable auxiliary x2 adapter and trainable output head.

یا به فارسی:

> بک‌بون اصلی DiT ثابت نگه داشته می‌شود و فقط یک adapter جانبی مبتنی بر مسیر x2، همراه با لایه‌ی خروجی، آموزش داده می‌شود.

## چرا این روش PEFT محسوب می‌شود؟

این روش PEFT است چون فقط بخش کوچکی از کل پارامترهای مدل آموزش داده می‌شود. در لاگ‌های آموزشی پروژه، تعداد پارامترهای trainable حدودا به صورت زیر گزارش شده است:

```text
17,656,864 / 692,749,600 = 2.55%
```

یعنی فقط حدود `2.55%` از پارامترهای کل مدل آموزش داده شده‌اند. این نسبت برای یک روش PEFT قابل دفاع است، چون بیشتر ظرفیت مدل از pretrained DiT گرفته می‌شود و فقط بخش کوچکی برای تطبیق با هدف جدید آموزش می‌بیند.

## تفاوت با full fine-tuning

در full fine-tuning همه‌ی پارامترهای DiT، شامل embedding ها، attention ها، MLP ها، conditioning ها و لایه‌ی خروجی، آپدیت می‌شوند. این کار پرهزینه‌تر است و می‌تواند باعث overfitting یا فراموشی دانش pretrained شود.

در روش فعلی:

- backbone اصلی ثابت است.
- فقط adapter جانبی و لایه‌ی خروجی train می‌شوند.
- تعداد پارامترهای trainable بسیار کمتر است.
- checkpoint مربوط به بخش trainable می‌تواند بسیار کوچک‌تر از checkpoint کامل باشد.

## تفاوت با LoRA

این روش نباید به عنوان LoRA معرفی شود. در LoRA معمولا داخل لایه‌های خطی attention یا MLP، ماتریس‌های کم‌رتبه‌ی قابل آموزش اضافه می‌شوند و وزن اصلی freeze می‌ماند.

در این پروژه، روش PEFT به شکل زیر است:

- اضافه کردن مسیر جانبی `x2`
- آموزش embedding و block مربوط به مسیر جانبی
- حفظ transformer اصلی DiT به صورت frozen
- آموزش `final_layer` برای تطبیق خروجی

بنابراین نام دقیق‌تر برای روش:

- `x2 Adapter Fine-Tuning`
- `Side-Stream Adapter Tuning`
- `Auxiliary Branch PEFT for DiT`
- `Frozen DiT Backbone with Trainable x2 Adapter`

## نکته‌ی مهم درباره‌ی final_layer

در پیاده‌سازی فعلی، `final_layer` هم train می‌شود. بنابراین بهتر است روش به عنوان **x2-only fine-tuning خالص** معرفی نشود. چون بخشی از بهبود احتمالی می‌تواند از آموزش لایه‌ی خروجی بیاید.

برای بیان دقیق‌تر در پایان‌نامه، بهتر است گفته شود:

> We freeze the pretrained DiT backbone and fine-tune only the auxiliary x2 branch and the final prediction layer.

این جمله دقیق‌تر از این است که گفته شود فقط x2 branch آموزش داده شده است.

## تابع هدف آموزشی

تابع هدف diffusion تغییر نکرده است. مدل همچنان با objective اصلی DiT آموزش داده می‌شود؛ یعنی مدل نویز اضافه‌شده به latent را پیش‌بینی می‌کند و loss اصلی بر اساس MSE محاسبه می‌شود. تفاوت روش در این است که gradient فقط به بخش‌های trainable adapter و `final_layer` می‌رسد.

پس contribution اصلی روش در objective نیست، بلکه در **محدود کردن پارامترهای قابل آموزش** و اضافه کردن یک مسیر کم‌پارامتر برای adaptation است.

## آموزش روی subset از کلاس‌ها

اسکریپت fine-tuning امکان آموزش روی subset مشخصی از کلاس‌های ImageNet را دارد. نکته‌ی خوب این است که label های اصلی ImageNet حفظ می‌شوند و کلاس‌ها remap نمی‌شوند. این باعث می‌شود embedding کلاس‌های pretrained همچنان با index اصلی خود استفاده شوند.

این طراحی برای PEFT مناسب است، چون مدل روی تعداد محدودی کلاس تطبیق داده می‌شود، بدون اینکه ساختار class-conditioning اصلی DiT تغییر کند.

## ادعای پیشنهادی برای پایان‌نامه

یک بیان دقیق و قابل دفاع برای روش:

> در این پژوهش، یک روش PEFT برای DiT معرفی شده است که در آن backbone اصلی pretrained DiT freeze می‌شود و فقط یک شاخه‌ی جانبی x2 به همراه لایه‌ی خروجی آموزش داده می‌شود. این روش با آموزش حدود 2.55 درصد از کل پارامترها، امکان تطبیق مدل با کلاس‌های هدف را فراهم می‌کند و در عین حال دانش مدل pretrained را تا حد زیادی حفظ می‌کند.

بیان انگلیسی پیشنهادی:

> We propose an adapter-style PEFT strategy for DiT, where the pretrained DiT backbone is frozen and only an auxiliary x2 side branch together with the final prediction layer is fine-tuned. This updates approximately 2.55% of the total parameters, reducing training cost while preserving the pretrained generative prior.

## ablation های لازم برای دفاع از روش PEFT

برای اینکه مشخص شود بهبود واقعا از PEFT پیشنهادی می‌آید، این آزمایش‌ها پیشنهاد می‌شوند:

1. مدل اصلی DiT بدون fine-tuning.
2. فقط `final_layer` قابل آموزش باشد.
3. فقط مسیر `x2` قابل آموزش باشد و `final_layer` freeze بماند.
4. مسیر `x2` همراه با `final_layer` قابل آموزش باشد، یعنی روش فعلی.
5. full fine-tuning با تعداد step محدود، برای مقایسه‌ی هزینه و کیفیت.

مهم‌ترین مقایسه بین حالت‌های 2، 3 و 4 است. اگر حالت 4 بهتر از حالت 2 باشد، می‌توان نتیجه گرفت که adapter جانبی `x2` واقعا ارزش افزوده دارد و بهبود فقط به خاطر train شدن `final_layer` نیست.

## محدودیت‌ها

روش فعلی چند محدودیت دارد که باید شفاف گزارش شوند:

- چون `final_layer` train می‌شود، روش x2-only خالص نیست.
- اگر فقط روی تعداد کمی تصویر آموزش داده شود، خطر overfitting وجود دارد.
- برای اثبات حفظ دانش pretrained، باید عملکرد روی کلاس‌های خارج از subset آموزشی هم بررسی شود.
- loss آموزشی به تنهایی برای قضاوت کافی نیست؛ باید کیفیت نمونه‌ها و metric هایی مثل FID/KID با تعداد نمونه‌ی کافی گزارش شوند.

## مقالات نزدیک به روش پیشنهادی

چند کار قبلی در diffusion models ایده‌ی مشابهی را دنبال کرده‌اند: نگه داشتن مدل pretrained به صورت frozen و آموزش دادن یک بخش کوچک‌تر، adapter، یا شاخه‌ی جانبی برای کنترل یا تطبیق مدل.

نزدیک‌ترین کارها به روش حاضر:

- **T2I-Adapter**: این مقاله مدل text-to-image اصلی را freeze می‌کند و adapter های سبک‌وزن را برای اتصال دانش داخلی مدل به سیگنال‌های کنترلی خارجی آموزش می‌دهد. از نظر فلسفه‌ی PEFT، بسیار نزدیک به روش حاضر است، چون مدل اصلی حفظ می‌شود و فقط adapter ها train می‌شوند.
- **ControlNet**: این مقاله یک شبکه‌ی trainable را در کنار مدل diffusion اصلی قرار می‌دهد و مدل اصلی را locked/frozen نگه می‌دارد. شباهت اصلی با روش حاضر در استفاده از یک مسیر جانبی trainable برای کنترل مدل pretrained است.
- **GLIGEN**: در این روش، همه‌ی وزن‌های مدل pretrained freeze می‌شوند و اطلاعات grounding از طریق لایه‌های جدید trainable و gated وارد مدل می‌شود. این ایده از نظر inject کردن اطلاعات جدید بدون fine-tune کامل backbone به روش حاضر نزدیک است.
- **Uni-ControlNet**: این کار روی مدل pretrained frozen فقط دو adapter اضافی را fine-tune می‌کند تا چند نوع کنترل محلی و سراسری را پشتیبانی کند. این مقاله برای دفاع از adapter-based PEFT روی diffusion models مفید است.
- **IP-Adapter**: این مقاله یک adapter سبک برای image prompt به مدل text-to-image اضافه می‌کند و diffusion model اصلی را freeze نگه می‌دارد. از نظر استفاده از adapter قابل آموزش روی backbone ثابت، رفرنس مناسبی است.
- **DiffFit**: این مقاله از side branch استفاده نمی‌کند، اما برای ادعای کلی PEFT در diffusion models مهم است، چون نشان می‌دهد با train کردن درصد بسیار کمی از پارامترها می‌توان مدل diffusion pretrained را به domain جدید منتقل کرد.

## جمع‌بندی

روش PEFT استفاده‌شده در این پروژه یک روش کم‌پارامتر و قابل دفاع برای تطبیق DiT است. نقطه‌ی قوت آن این است که backbone اصلی pretrained حفظ می‌شود و فقط حدود 2.55 درصد پارامترها آموزش داده می‌شوند. با این حال، برای گزارش علمی دقیق، باید روش به عنوان **trainable x2 adapter plus output head over a frozen DiT backbone** معرفی شود، نه LoRA و نه x2-only fine-tuning خالص.

## References

1. Zhang, L., Rao, A., & Agrawala, M. (2023). **Adding Conditional Control to Text-to-Image Diffusion Models**. arXiv:2302.05543. [https://arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543)

2. Mou, C., Wang, X., Xie, L., Wu, Y., Zhang, J., Qi, Z., Shan, Y., & Qie, X. (2023). **T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models**. arXiv:2302.08453. [https://arxiv.org/abs/2302.08453](https://arxiv.org/abs/2302.08453)

3. Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., Li, C., & Lee, Y. J. (2023). **GLIGEN: Open-Set Grounded Text-to-Image Generation**. arXiv:2301.07093. [https://arxiv.org/abs/2301.07093](https://arxiv.org/abs/2301.07093)

4. Zhao, S., Chen, D., Chen, Y.-C., Bao, J., Hao, S., Yuan, L., & Wong, K.-Y. K. (2023). **Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models**. arXiv:2305.16322. [https://arxiv.org/abs/2305.16322](https://arxiv.org/abs/2305.16322)

5. Ye, H., Zhang, J., Liu, S., Han, X., & Yang, W. (2023). **IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models**. arXiv:2308.06721. [https://arxiv.org/abs/2308.06721](https://arxiv.org/abs/2308.06721)

6. Xie, E., Yao, L., Shi, H., Liu, Z., Zhou, D., Liu, Z., Li, J., & Li, Z. (2023). **DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning**. arXiv:2304.06648. [https://arxiv.org/abs/2304.06648](https://arxiv.org/abs/2304.06648)

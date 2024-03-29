# CausalDiff
According to the advice of Reviewer NXAq, we generated images 

We first reproduced the results in Figure 8 to visualize what the model has learned in $s$ and $z.
![GitHub set up](https://github.com/ZhangMingKun1/CausalDiff/blob/main/generated_images/fig_cases.png "Generated images of cases")
</center> <!--结束居中对齐-->


Moreover, we plot the images generated conditioned on $s$ and $z$ respectively, with interpolation from cat to dog, demonstrating the semantic transition from cat to dog. The **evolution process on $s$ shows that the semantics initially fading away are cat-related, such as the facial features of a cat, while retaining the animal's body, a common feature between cats and dogs**. This is followed by a transition towards a dog, marked by the appearance of canine facial features. On the other hand, the evolution process of $z$ involves some meaningless information. This evolution process aligns with human cognition as expected.

![GitHub set up](https://github.com/ZhangMingKun1/CausalDiff/blob/main/generated_images/fig_interpolation_CausalDiff.png "Generated images resulting from the interpolation using CausalDiff")
</center> <!--结束居中对齐-->


Additionally, we visualized the generated images of $p_{\theta}(x|s_{\text{cat}}, z_{\text{dog}})$ and $p_{\theta}(x|s_{\text{dog}}, z_{\text{cat}})$ to demonstrate the impact of $s$ and $z$ during the generation process. We observed that the **semantics of the generated images are primarily controlled by $s$**, but $z$ brings some disturbance since it is derived from a different image. This is **in line with the control mechanism we introduced in Section 4.1**.


![GitHub set up](https://github.com/ZhangMingKun1/CausalDiff/blob/main/generated_images/fig_interpolation_encoder.png "Generated images resulting from the interpolation using the encoder of CausalDiff")
</center> <!--结束居中对齐-->

Regarding the latent factors $s$ and $z$ inference through different methods, we utilized **(a)** the inference approach of Eq. 10, **(b)** directly obtained $s$ and $z$ using the encoder, and **(c)** inferred $s$ and $z$ using a pretrained CausalDiff through Eq. 10. We found that the **results from Eq. 10 and the encoder are similar**, aligning with the analysis conclusion in A8 (the encoder's inference method has comparable accuracy but poor robustness, indicating that the encoder performs well on clean samples but may be inaccurate at $x + \delta$ ). Furthermore, we discovered that the model of pretrained CausalDiff (without joint training) **contained identical information in $s$ and $z$**. This is because, without the guidance of a discriminator, the model fails to differentiate and learn the distinct roles of $s$ and $z$, resulting in them holding consistent information.

![GitHub set up](https://github.com/ZhangMingKun1/CausalDiff/blob/main/generated_images/fig_interpolation_pretrain.png "Generated images resulting from the interpolation using the pretrained CausalDiff")
</center> <!--结束居中对齐-->

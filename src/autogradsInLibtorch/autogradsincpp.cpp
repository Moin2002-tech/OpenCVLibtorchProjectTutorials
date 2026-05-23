//
// Created by moinshaikh on 3/7/26.
//
/*
* The LibTorch Explorer's Cheat Sheet

    Memory Management: Always distinguish between Storage (raw data) and Tensor Metadata (shape/stride).
    Use empty for speed, zeros for initialization, and from_blob for mapping existing memory.

    The "Underscore" Rule: Methods without an underscore (e.g., clamp()) are functional and create new copies.
    Methods with an underscore (e.g., clamp_()) are mutable and act directly on the storage—use these to save VRAM on your T600.

    Autograd Logic: Everything is just a node in a graph. requires_grad(true) is the "Record" button,
    backward() is the "Calculate Derivatives" trigger, and .grad() is the "Result" bucket.

    Calculus Under the Hood: Never be intimidated by backpropagation.
    It is literally just the Power Rule and Chain Rule from 11th/12th grade, executed across thousands of nodes in parallel.
 */
#include<torch/torch.h>
#include"../../external/third_party/doctest.hpp"
using std::cout;

torch::Tensor binaryCrossEntropyLoss(torch::Tensor &pridiction,torch::Tensor &target)
{
    auto epislon =  1e-8;
    pridiction = torch::clamp(pridiction,epislon,1-epislon);
    return -(target * torch::log(pridiction)+ (1- target) * torch::log(1 - pridiction));
}

template<typename T>
T print(T t)
{
    std::cout<<t;
}

TEST_CASE("autoGradinC++")
{

    /**
    * Bridging 11th-Grade Calculus to Autograd

    When you write y = x.pow(2) in C++, the Autograd engine essentially creates a "node" in a graph that looks like this:

    Function: f(x)=xn

    Derivative: f′(x)=n⋅xn−1

    Why it feels different in Code

    In school, you solved dxdy​ manually on paper.
    In LibTorch, the engine does two things you didn't have to worry about:

    Numerical Storage: It doesn't just know the formula; it remembers the actual value of x used at that specific moment (the "Forward Pass" value).

    The Chain Rule Integration: If your formula were y=(3x2+5)3, LibTorch doesn't force you to expand the polynomial. It breaks it into tiny pieces, solves the derivative for each piece using the power rule, and multiplies them together—exactly like you did with the chain rule in class.

    The "Dry Run" Logic

        When you call y.backward():

            The Engine sees the graph: It finds the pow node.

            It applies the Power Rule: n⋅xn−1.

            It plugs in your data: If x=3 and n=2, it calculates 2⋅32−1=6.

            It saves the result: It places 6 into the .grad() container.
     */



    auto x = torch::tensor(3.0,torch::requires_grad(true));
    auto y = x.pow(2);

    std::cout<<"values of X: "<<"\n"<<x;
    cout<<"values of Y: "<<"\n"<< y;

    y.backward();
    std::cout<<y<<":  afterBackward ";
    std::cout<<"the value of x: "<<x.grad();


    //
    /*
    * 2. Intermediate Tensors do not store .grad()

    You wrote:
    C++

    y1.grad(); // This will return an empty tensor

    In LibTorch, only "Leaf" tensors store gradients. Leaf tensors are the ones you created yourself (like x1). Intermediate tensors created during math (like y1) are discarded by default to save memory. If you really need the gradient of an intermediate node, you must call .retain_grad() on it before running .backward().
    3. Missing optimizer.zero_grad() / Accumulated Gradients

    You are calling .backward() twice. Because LibTorch accumulates gradients, if you run the same code again, the .grad will keep adding up. Always clear your buffers if you are testing iteratively.
    Corrected Code Pattern

    Here is the clean way to execute your math with the Chain Rule:
    C++

    // 1. Correct graph connectivity
    auto x1 = torch::tensor(4.0, torch::requires_grad(true));
    auto y1 = x1.pow(2); // y1 = x^2
    auto z1 = torch::sin(y1); // z1 = sin(x^2)

    // To see the gradient of y1, you MUST ask for it before backward
    y1.retain_grad();

    z1.backward();

    std::cout << "Gradient of x1 (dz/dx): " << x1.grad() << "\n";
    std::cout << "Gradient of y1 (dz/dy): " << y1.grad() << "\n";

    The Math Check (Chain Rule)

        Function: z=sin(x2)

        dydz​ (where y=x2): cos(y)=cos(x2)=cos(16)≈−0.952

        dxdy​: 2x=8

        Total dxdz​ (Chain Rule): cos(x2)⋅2x≈−0.952⋅8≈−7.619
     */
    auto x1 = torch::tensor(4.0,torch::requires_grad(true));
    auto y1 = x1.pow(2);
    auto z1 =  torch::sin(y1);


    cout<<x1;
    cout<<y1;
    cout<<z1;

    y1.retain_grad();
    z1.backward();

  std::cout<< x1.grad();
   //std::cout<< y1.grad();


    auto inputFeature = torch::tensor(6.7);
    auto trueLable = torch::tensor(0.0);
    auto weight = torch::tensor(1.0);
    auto bias = torch::tensor(0.0);

    auto WeightedSum = weight *  inputFeature + bias; //weighted SUm linear
    auto Ypred=  torch::sigmoid(WeightedSum);
    auto loss = binaryCrossEntropyLoss(Ypred,trueLable);


    std::cout<<"the loss of predicted = "<<loss;

    //Derivative
    //dloss/d(y_pred) =  Loss with respect to prediction ypred
    auto dLoss_dy_pred=  (Ypred - trueLable)/(Ypred*  (1 - Ypred));

    auto predy_dz =Ypred *( 1-Ypred);
    auto dz_dw = x;
    auto dz_db = torch::tensor(1);

    auto DL_dw =dLoss_dy_pred * predy_dz * dz_dw;
    auto DL_db =dLoss_dy_pred * predy_dz * dz_db;

    std::cout<<"Manual Gradient of loss w.r.t weight (dw) =  "<<DL_dw;
    std::cout<<"Manual Gradient of loss w.r.t bias (db) =  "<<DL_db;


    auto x2 =  torch::tensor(6.7);
    auto y2 = torch::tensor(0.0);

    auto weight1 =  torch::tensor(1.0,torch::requires_grad(true));
    auto bias1 =  torch::tensor(0.0, torch::requires_grad(true));

    print(weight1);
    print(bias1);

    auto z2 = weight1  * x2 + bias1;
    print(z2);

   auto Yprediction = torch::sigmoid(z2);
    print(Yprediction);

    auto loss2 = binaryCrossEntropyLoss(Yprediction,y2);
    print(loss2);


    loss2.backward();

    auto grad1 = weight1.grad();
    auto grad2 =  bias1.grad();

    print(grad1);
    print(grad2);









}